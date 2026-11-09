/**
 * BreatheFlow AI - Presage SmartSpectra SDK Integration Service
 * 
 * Provides camera-derived respiration and pulse sensing using the official
 * @smartspectra/node-sdk package within an Electron runtime.
 */

import { decodeMetrics } from '@smartspectra/node-sdk/messages';
import {
  SmartSpectraSDK,
  breathingMetrics,
  cardioMetrics,
  ValidationCode,
  ProcessingStatus,
} from '@smartspectra/node-sdk/renderer';

export interface VitalsReading {
  breathingRateBpm: number | null;
  breathingConfidence: number | null;
  breathingStable: boolean;
  pulseRateBpm: number | null;
  pulseConfidence: number | null;
  pulseStable: boolean;
  validationStatus: string;
  validationCode: number;
  processingStatus: string;
  timestampUs: number;
}

export type VitalsListener = (vitals: VitalsReading) => void;
export type StreamListener = (stream: MediaStream) => void;
export type StatusListener = (statusText: string) => void;

function getValidationMessage(code: number, hint?: string): string {
  if (hint && hint.trim().length > 0) return hint;
  switch (code) {
    case ValidationCode.kOk:
      return 'Optimal signal: Face and chest aligned.';
    case ValidationCode.kNoFaceFound:
      return 'Position your face in front of the camera.';
    case ValidationCode.kMultipleFacesFound:
      return 'Multiple faces detected. Please ensure only one person is visible.';
    case ValidationCode.kFaceNotCentered:
      return 'Center your face in the camera frame.';
    case ValidationCode.kTooDark:
      return 'Environment is too dark. Increase lighting.';
    case ValidationCode.kTooBright:
      return 'Environment is too bright or has high glare.';
    case ValidationCode.kChestNotVisible:
      return 'Position camera so upper chest and shoulders are visible.';
    case ValidationCode.kCameraTuning:
      return 'Camera adjusting exposure and focus...';
    case ValidationCode.kFrameRateTooLow:
      return 'Camera frame rate is below recommended threshold.';
    case ValidationCode.kExcessiveMotion:
      return 'Excessive movement detected. Please remain still.';
    case ValidationCode.kFaceTooClose:
      return 'Move back slightly from the camera.';
    case ValidationCode.kFaceTooFar:
      return 'Move closer to the camera.';
    case ValidationCode.kFaceNotForward:
      return 'Face forward toward the camera.';
    default:
      return 'Calibrating camera sensor...';
  }
}

function getProcessingStatusMessage(status: number): string {
  switch (status) {
    case ProcessingStatus.kUninitialized:
      return 'Uninitialized';
    case ProcessingStatus.kIdle:
      return 'Idle';
    case ProcessingStatus.kStarting:
      return 'Starting SmartSpectra pipeline...';
    case ProcessingStatus.kRunning:
      return 'Active Sensing';
    case ProcessingStatus.kStopping:
      return 'Stopping...';
    case ProcessingStatus.kError:
      return 'Error in processing';
    default:
      return 'Ready';
  }
}

export class SmartSpectraController {
  private sdk: SmartSpectraSDK | null = null;
  private vitalsListeners: Set<VitalsListener> = new Set();
  private streamListeners: Set<StreamListener> = new Set();
  private statusListeners: Set<StatusListener> = new Set();
  private currentVitals: VitalsReading = {
    breathingRateBpm: null,
    breathingConfidence: null,
    breathingStable: false,
    pulseRateBpm: null,
    pulseConfidence: null,
    pulseStable: false,
    validationStatus: 'Initializing',
    validationCode: -1,
    processingStatus: 'Idle',
    timestampUs: 0,
  };

  public isAvailable(): boolean {
    return typeof window !== 'undefined' && Boolean((window as unknown as { __smartspectraBridge?: unknown }).__smartspectraBridge);
  }

  public subscribeVitals(listener: VitalsListener): () => void {
    this.vitalsListeners.add(listener);
    listener(this.currentVitals);
    return () => this.vitalsListeners.delete(listener);
  }

  public subscribeStream(listener: StreamListener): () => void {
    this.streamListeners.add(listener);
    return () => this.streamListeners.delete(listener);
  }

  public subscribeStatus(listener: StatusListener): () => void {
    this.statusListeners.add(listener);
    return () => this.statusListeners.delete(listener);
  }

  private notifyVitals(): void {
    for (const listener of this.vitalsListeners) {
      try {
        listener({ ...this.currentVitals });
      } catch (err) {
        console.error('Error in vitals listener:', err);
      }
    }
  }

  private notifyStatus(msg: string): void {
    for (const listener of this.statusListeners) {
      try {
        listener(msg);
      } catch (err) {
        console.error('Error in status listener:', err);
      }
    }
  }

  public async start(): Promise<void> {
    if (!this.isAvailable()) {
      throw new Error(
        'SmartSpectra bridge unavailable. Please launch BreatheFlow inside the Electron desktop runtime (`npm run electron:dev`).'
      );
    }

    if (this.sdk) {
      await this.stop();
    }

    this.notifyStatus('Initializing SmartSpectra SDK...');

    // Instantiate official SmartSpectraSDK; authentication credential is automatically
    // provisioned by Electron main process from developer .env
    this.sdk = new SmartSpectraSDK({
      requestedMetrics: [...breathingMetrics, ...cardioMetrics],
      enableAccumulatedOutput: true,
      enableTelemetry: false,
    });

    this.sdk.on('streamAvailable', (stream: MediaStream) => {
      for (const listener of this.streamListeners) {
        try {
          listener(stream);
        } catch (err) {
          console.error('Error in stream listener:', err);
        }
      }
    });

    this.sdk.on('processingStatus', (status) => {
      this.currentVitals.processingStatus = getProcessingStatusMessage(status);
      this.notifyStatus(this.currentVitals.processingStatus);
      this.notifyVitals();
    });

    this.sdk.on('validationStatus', (code, ts, hint) => {
      this.currentVitals.validationCode = code;
      this.currentVitals.validationStatus = getValidationMessage(code, hint);
      this.currentVitals.timestampUs = ts;
      this.notifyVitals();
    });

    this.sdk.on('metrics', (buf: Uint8Array, ts: number) => {
      try {
        const decoded = decodeMetrics(buf);
        if (decoded && decoded.breathing) {
          const rates = decoded.breathing.rate;
          if (rates && rates.length > 0) {
            const latest = rates[rates.length - 1];
            if (latest.value !== null && latest.value !== undefined) {
              this.currentVitals.breathingRateBpm = Math.round(latest.value * 10) / 10;
              this.currentVitals.breathingConfidence = latest.confidence ? Math.round(latest.confidence) : null;
              this.currentVitals.breathingStable = Boolean(latest.stable);
            }
          }
        }

        if (decoded && decoded.cardio) {
          const pulses = decoded.cardio.pulseRate;
          if (pulses && pulses.length > 0) {
            const latestPulse = pulses[pulses.length - 1];
            if (latestPulse.value !== null && latestPulse.value !== undefined) {
              this.currentVitals.pulseRateBpm = Math.round(latestPulse.value);
              this.currentVitals.pulseConfidence = latestPulse.confidence ? Math.round(latestPulse.confidence) : null;
              this.currentVitals.pulseStable = Boolean(latestPulse.stable);
            }
          }
        }

        this.currentVitals.timestampUs = ts;
        this.notifyVitals();
      } catch (err) {
        console.warn('Failed to decode SmartSpectra metrics packet:', err);
      }
    });

    this.sdk.on('error', (code, message, retryable) => {
      const errStr = `SmartSpectra Error (${code}): ${message}${retryable ? ' (retryable)' : ''}`;
      console.error(errStr);
      this.notifyStatus(errStr);
    });

    await this.sdk.start();
    this.notifyStatus('SmartSpectra camera sensor running.');
  }

  public async stop(): Promise<void> {
    if (!this.sdk) return;
    this.notifyStatus('Stopping camera sensor...');
    try {
      await this.sdk.stop();
      this.sdk.destroy();
    } catch (err) {
      console.warn('Error during SmartSpectra shutdown:', err);
    } finally {
      this.sdk = null;
      this.currentVitals.processingStatus = 'Idle';
      this.notifyVitals();
      this.notifyStatus('Camera sensor stopped.');
    }
  }
}

export const smartSpectraController = new SmartSpectraController();
