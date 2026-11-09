import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Mic,
  Square,
  Play,
  Pause,
  RotateCcw,
  Volume2,
  Wind,
  Sparkles,
  AlertCircle,
  CheckCircle2,
  UploadCloud,
  Activity,
  Layers,
  StopCircle,
  PlayCircle,
  Sparkle,
} from 'lucide-react';
import { AudioRecorder, type RecordedAudio } from './utils/audioRecorder';
import { SmartSpectraCard } from './components/SmartSpectraCard';

interface SuggestedProtocol {
  action?: string;
  protocol: string;
  inhale_s?: number;
  hold_in_s?: number;
  exhale_s?: number;
  hold_out_s?: number;
  cycles?: number;
}

interface ControllerTick {
  phase: string;
  current_cycle: number;
  total_cycles: number;
  phase_remaining_s: number;
  phase_duration_s: number;
}

interface VoiceTurnResponse {
  status: string;
  transcript: string;
  spoken_response: string;
  suggested_protocol: SuggestedProtocol | null;
  controller_tick: ControllerTick | null;
  audio_base64: string | null;
  audio_format: string;
  phase_cues?: Record<string, string> | null;
  is_fallback: boolean;
  error_message: string | null;
}

type BreathingPhaseType = 'idle' | 'begin' | 'inhale' | 'hold_in' | 'exhale' | 'hold_out' | 'complete';

export function App() {
  // Turn & Audio Recording State
  const [isRecording, setIsRecording] = useState(false);
  const [recordDuration, setRecordDuration] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string>('Ready to listen. Click to speak naturally.');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Turn Results
  const [transcript, setTranscript] = useState<string | null>(null);
  const [assistantText, setAssistantText] = useState<string | null>(null);
  const [isFallback, setIsFallback] = useState(false);
  const [protocol, setProtocol] = useState<SuggestedProtocol | null>(null);
  const [introAudioUrl, setIntroAudioUrl] = useState<string | null>(null);
  const [isPlayingIntroAudio, setIsPlayingIntroAudio] = useState(false);

  // Voice Cues Cache (Phase audio clips: begin, inhale, hold, exhale, complete)
  const [phaseCues, setPhaseCues] = useState<Record<string, string> | null>(null);
  const audioElementsRef = useRef<Map<string, HTMLAudioElement>>(new Map());

  // Real-Time Guided Breathing State
  const [isPacingActive, setIsPacingActive] = useState(false);
  const [currentCycle, setCurrentCycle] = useState(1);
  const [targetCycles, setTargetCycles] = useState(4);
  const [activePhase, setActivePhase] = useState<BreathingPhaseType>('idle');
  const [phaseRemaining, setPhaseRemaining] = useState(0);
  const [phaseTotalDuration, setPhaseTotalDuration] = useState(0);
  const [pacingProgress, setPacingProgress] = useState(0); // 0.0 to 1.0

  // Presage SmartSpectra Camera Cadence State
  const [measuredCadence, setMeasuredCadence] = useState<number | null>(null);

  // Configuration Settings
  const [useMockMode, setUseMockMode] = useState(false);
  const [selectedBrain, setSelectedBrain] = useState<'gemini' | 'ollama'>('gemini');

  // References
  const recorderRef = useRef<AudioRecorder | null>(null);
  const timerIntervalRef = useRef<number | null>(null);
  const introAudioRef = useRef<HTMLAudioElement | null>(null);
  const activePacingTimeoutRef = useRef<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const isStoppingPacingRef = useRef(false);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
      if (activePacingTimeoutRef.current) clearTimeout(activePacingTimeoutRef.current);
      if (introAudioUrl) URL.revokeObjectURL(introAudioUrl);
      stopAllAudio();
    };
  }, [introAudioUrl]);

  // Preload Voice Cues into HTMLAudioElements
  const preloadAudioCues = useCallback((cues: Record<string, string>) => {
    // Stop and clear previous elements
    audioElementsRef.current.forEach((audio) => {
      audio.pause();
    });
    audioElementsRef.current.clear();

    Object.entries(cues).forEach(([key, b64]) => {
      if (!b64) return;
      try {
        const audio = new Audio(`data:audio/wav;base64,${b64}`);
        audio.preload = 'auto';
        audioElementsRef.current.set(key, audio);
      } catch (e) {
        console.warn(`Could not preload cue '${key}':`, e);
      }
    });
  }, []);

  // Play a specific voice cue without overlap
  const playVoiceCue = useCallback((cueKey: string) => {
    if (isStoppingPacingRef.current) return;

    // Pause intro audio if playing
    if (introAudioRef.current) {
      introAudioRef.current.pause();
      setIsPlayingIntroAudio(false);
    }

    // Stop currently playing cue elements
    audioElementsRef.current.forEach((audio) => {
      audio.pause();
      audio.currentTime = 0;
    });

    const cue = audioElementsRef.current.get(cueKey);
    if (cue) {
      cue.currentTime = 0;
      cue.play().catch((err) => {
        // Autoplay policy or user cancellation
        console.warn(`Cue playback for '${cueKey}' halted:`, err);
      });
    }
  }, []);

  const stopAllAudio = () => {
    if (introAudioRef.current) {
      introAudioRef.current.pause();
      introAudioRef.current.currentTime = 0;
      setIsPlayingIntroAudio(false);
    }
    audioElementsRef.current.forEach((audio) => {
      audio.pause();
      audio.currentTime = 0;
    });
  };

  // Start Voice Recording
  const handleStartRecording = async () => {
    stopPacingSession();
    setErrorMessage(null);
    try {
      const recorder = new AudioRecorder();
      recorderRef.current = recorder;
      await recorder.start();

      setIsRecording(true);
      setRecordDuration(0);
      setStatusMessage('Listening... describe your current state or breathing needs.');

      const startTime = Date.now();
      timerIntervalRef.current = window.setInterval(() => {
        setRecordDuration(Math.floor((Date.now() - startTime) / 1000));
      }, 500);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Microphone permission was denied or is unavailable.';
      setErrorMessage(msg);
      setStatusMessage('Microphone access failed.');
      setIsRecording(false);
    }
  };

  // Stop Recording and Process Turn
  const handleStopRecording = async () => {
    if (!recorderRef.current || !isRecording) return;

    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }

    setIsRecording(false);
    setIsProcessing(true);
    setStatusMessage('Transcribing speech & analyzing emotion...');

    try {
      const { blob }: RecordedAudio = await recorderRef.current.stop();
      recorderRef.current = null;

      await uploadAndProcessAudio(blob, 'microphone_turn.wav');
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Error capturing or stopping recording.';
      setErrorMessage(msg);
      setIsProcessing(false);
      setStatusMessage('Turn failed.');
    }
  };

  // Upload Audio Blob to FastAPI Backend
  const uploadAndProcessAudio = async (audioBlob: Blob, filename = 'turn.wav') => {
    setIsProcessing(true);
    setErrorMessage(null);
    setStatusMessage('Connecting to BreatheFlow AI voice pipeline...');

    const formData = new FormData();
    formData.append('file', audioBlob, filename);
    formData.append('mock', useMockMode ? 'true' : 'false');
    formData.append('brain', selectedBrain);

    try {
      const res = await fetch('/api/voice-turn', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        let errDetail = `Server responded with status ${res.status}`;
        try {
          const errData = await res.json();
          if (errData && errData.detail) errDetail = errData.detail;
        } catch {
          // Keep generic detail
        }
        throw new Error(errDetail);
      }

      const data: VoiceTurnResponse = await res.json();

      setTranscript(data.transcript);
      setAssistantText(data.spoken_response);
      setIsFallback(data.is_fallback);
      setProtocol(data.suggested_protocol);

      if (data.suggested_protocol && data.suggested_protocol.cycles) {
        setTargetCycles(data.suggested_protocol.cycles);
      }

      // Preload phase voice cues
      if (data.phase_cues) {
        setPhaseCues(data.phase_cues);
        preloadAudioCues(data.phase_cues);
      }

      // Handle returned Kaveri initial response audio
      if (data.audio_base64) {
        const byteCharacters = atob(data.audio_base64);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const soundBlob = new Blob([byteArray], { type: 'audio/wav' });

        if (introAudioUrl) URL.revokeObjectURL(introAudioUrl);
        const newUrl = URL.createObjectURL(soundBlob);
        setIntroAudioUrl(newUrl);

        setStatusMessage('Kaveri voice response playing. Breathing session will begin shortly...');
        playIntroAudio(newUrl, data.suggested_protocol, data.phase_cues || null);
      } else {
        setStatusMessage('Turn completed.');
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to communicate with BreatheFlow backend.';
      setErrorMessage(msg);
      setStatusMessage('Voice turn failed.');
    } finally {
      setIsProcessing(false);
    }
  };

  // Play Initial Response Audio with seamless transition to Guided Breathing
  const playIntroAudio = (
    urlToPlay?: string,
    protoToStart?: SuggestedProtocol | null,
    cuesAvailable?: Record<string, string> | null
  ) => {
    const targetUrl = urlToPlay || introAudioUrl;
    if (!targetUrl) return;

    if (introAudioRef.current) {
      introAudioRef.current.pause();
    }

    const audio = new Audio(targetUrl);
    introAudioRef.current = audio;

    audio.onplay = () => setIsPlayingIntroAudio(true);
    audio.onended = () => {
      setIsPlayingIntroAudio(false);
      // Automatically begin the voice-guided breathing session once Kaveri finishes speaking!
      const activeProto = protoToStart !== undefined ? protoToStart : protocol;
      if (activeProto) {
        setStatusMessage('Starting voice-guided breathing exercise...');
        startVoiceGuidedBreathingSession(activeProto, cuesAvailable || phaseCues);
      } else {
        setStatusMessage('Session ready.');
      }
    };
    audio.onerror = () => {
      setIsPlayingIntroAudio(false);
      setErrorMessage('Audio playback encountered an error.');
    };

    audio.play().catch(() => {
      setIsPlayingIntroAudio(false);
      setStatusMessage('Audio response ready. Click Start Breathing to begin.');
    });
  };

  const pauseIntroAudio = () => {
    if (introAudioRef.current) {
      introAudioRef.current.pause();
      setIsPlayingIntroAudio(false);
    }
  };

  // --- Voice-Guided Deterministic Breathing Session Engine ---
  const startVoiceGuidedBreathingSession = (
    selectedProto?: SuggestedProtocol | null,
    cues?: Record<string, string> | null
  ) => {
    const activeProto = selectedProto || protocol;
    if (!activeProto) return;

    // Stop any existing session
    stopPacingSession();
    isStoppingPacingRef.current = false;

    // If cues are provided, ensure they are loaded
    if (cues) {
      preloadAudioCues(cues);
    }

    setIsPacingActive(true);
    setStatusMessage('Breathe with Kaveri. Follow the spoken cues and glowing orb.');

    const inhaleS = activeProto.inhale_s || 4.0;
    const holdInS = activeProto.hold_in_s || 0.0;
    const exhaleS = activeProto.exhale_s || 4.0;
    const holdOutS = activeProto.hold_out_s || 0.0;
    const totalCyclesCount = targetCycles || activeProto.cycles || 4;

    // Step 0: "Let's begin." introduction cue
    setActivePhase('begin');
    setPhaseRemaining(2.0);
    setPhaseTotalDuration(2.0);
    playVoiceCue('begin');

    activePacingTimeoutRef.current = window.setTimeout(() => {
      if (isStoppingPacingRef.current) return;
      runCycle(1, totalCyclesCount, inhaleS, holdInS, exhaleS, holdOutS);
    }, 2000);
  };

  // Run a single cycle of the protocol
  const runCycle = (
    cycleNum: number,
    totalCyclesNum: number,
    inhaleS: number,
    holdInS: number,
    exhaleS: number,
    holdOutS: number
  ) => {
    if (isStoppingPacingRef.current) return;
    if (cycleNum > totalCyclesNum) {
      // All cycles completed successfully!
      finishSession();
      return;
    }

    setCurrentCycle(cycleNum);

    // Sequence of active phases for this cycle (skips 0-second holds!)
    const phaseSequence: Array<{ phase: BreathingPhaseType; duration: number; cue: string }> = [];

    if (inhaleS > 0) {
      phaseSequence.push({ phase: 'inhale', duration: inhaleS, cue: 'inhale' });
    }
    if (holdInS > 0) {
      phaseSequence.push({ phase: 'hold_in', duration: holdInS, cue: 'hold' });
    }
    if (exhaleS > 0) {
      phaseSequence.push({ phase: 'exhale', duration: exhaleS, cue: 'exhale' });
    }
    if (holdOutS > 0) {
      phaseSequence.push({ phase: 'hold_out', duration: holdOutS, cue: 'hold' });
    }

    // Execute phases sequentially
    executePhaseIndex(0, phaseSequence, () => {
      // Advance to next cycle
      runCycle(cycleNum + 1, totalCyclesNum, inhaleS, holdInS, exhaleS, holdOutS);
    });
  };

  // Execute a specific phase index in the sequence
  const executePhaseIndex = (
    index: number,
    sequence: Array<{ phase: BreathingPhaseType; duration: number; cue: string }>,
    onCycleDone: () => void
  ) => {
    if (isStoppingPacingRef.current) return;

    if (index >= sequence.length) {
      onCycleDone();
      return;
    }

    const current = sequence[index];
    setActivePhase(current.phase);
    setPhaseTotalDuration(current.duration);
    setPhaseRemaining(current.duration);
    setPacingProgress(0);

    // Speak Kaveri phase cue aloud
    playVoiceCue(current.cue);

    const phaseStartTime = performance.now();
    const durationMs = current.duration * 1000;

    // High-resolution interval for smooth countdown and orb scale
    const stepInterval = window.setInterval(() => {
      if (isStoppingPacingRef.current) {
        clearInterval(stepInterval);
        return;
      }

      const elapsedMs = performance.now() - phaseStartTime;
      const remainingSec = Math.max(0, (durationMs - elapsedMs) / 1000);
      const progress = Math.min(1.0, elapsedMs / durationMs);

      setPhaseRemaining(parseFloat(remainingSec.toFixed(1)));
      setPacingProgress(progress);

      if (elapsedMs >= durationMs) {
        clearInterval(stepInterval);
        // Advance to next phase in this cycle
        executePhaseIndex(index + 1, sequence, onCycleDone);
      }
    }, 50);
  };

  // Session Completed
  const finishSession = () => {
    setActivePhase('complete');
    setIsPacingActive(false);
    playVoiceCue('complete');
    setStatusMessage('Session complete. Take a gentle breath and notice how you feel.');
  };

  // Stop / Cancel Session
  const stopPacingSession = () => {
    isStoppingPacingRef.current = true;
    if (activePacingTimeoutRef.current) {
      clearTimeout(activePacingTimeoutRef.current);
      activePacingTimeoutRef.current = null;
    }
    stopAllAudio();
    setIsPacingActive(false);
    setActivePhase('idle');
    setPhaseRemaining(0);
    setStatusMessage('Session stopped. Click below whenever you wish to resume.');
  };

  // Handle Manual File Upload
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    uploadAndProcessAudio(file, file.name);
  };

  // Format Helper for Protocol Names
  const formatProtocolName = (name: string) => {
    switch (name.toLowerCase()) {
      case 'box_breathing':
        return 'Box Breathing (4-4-4-4)';
      case 'four_seven_eight':
      case '4_7_8':
        return '4-7-8 Relaxing Breath';
      case 'coherent_breathing':
        return 'Coherent Resonant Breathing (5-0-5-0)';
      case 'physiological_sigh':
        return 'Physiological Sigh (Acute Reset)';
      default:
        return name.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
    }
  };

  // Calculate dynamic orb scale and glow based on active breathing phase
  const getDynamicOrbStyle = () => {
    if (isPacingActive) {
      let scale = 1.0;
      let glow = 'rgba(20, 184, 166, 0.35)';

      if (activePhase === 'inhale') {
        // Smoothly expand from 1.0 to 1.48
        scale = 1.0 + pacingProgress * 0.48;
        glow = 'rgba(45, 212, 191, 0.6)';
      } else if (activePhase === 'hold_in') {
        // Hold full volume steady with luminous mint glow
        scale = 1.48;
        glow = 'rgba(94, 234, 212, 0.75)';
      } else if (activePhase === 'exhale') {
        // Smoothly contract from 1.48 back down to 1.0
        scale = 1.48 - pacingProgress * 0.48;
        glow = 'rgba(20, 184, 166, 0.45)';
      } else if (activePhase === 'hold_out') {
        // Hold resting volume
        scale = 1.0;
        glow = 'rgba(15, 118, 110, 0.4)';
      } else if (activePhase === 'begin') {
        scale = 1.05;
        glow = 'rgba(99, 102, 241, 0.5)';
      }

      return {
        transform: `scale(${scale})`,
        boxShadow: `0 0 80px ${glow}, inset 0 0 50px ${glow}`,
        transition: 'transform 0.08s linear',
      };
    }

    if (isRecording) {
      return {
        transform: 'scale(1.05)',
        boxShadow: '0 0 85px rgba(239, 68, 68, 0.55)',
      };
    }

    return {};
  };

  return (
    <div style={styles.container}>
      {/* Background Ambience */}
      <div style={styles.bgGlow} />

      {/* Top Header */}
      <header style={styles.header}>
        <div style={styles.brandRow}>
          <div style={styles.logoBadge}>
            <Wind size={22} color="#2dd4bf" />
          </div>
          <div>
            <h1 style={styles.title}>BreatheFlow AI</h1>
            <p style={styles.tagline}>Voice-First Adaptive Breathing & Wellness</p>
          </div>
        </div>

        {/* Engine Mode Controls */}
        <div style={styles.modeControls}>
          <div style={styles.badge}>
            <Activity size={14} color="#2dd4bf" />
            <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
              Engine: <strong style={{ color: '#2dd4bf' }}>{useMockMode ? 'MOCK' : 'LIVE'}</strong>
            </span>
          </div>

          <button
            onClick={() => setUseMockMode(!useMockMode)}
            style={{
              ...styles.toggleBtn,
              background: useMockMode ? 'rgba(234, 179, 8, 0.15)' : 'rgba(20, 184, 166, 0.15)',
              borderColor: useMockMode ? '#eab308' : '#14b8a6',
              color: useMockMode ? '#fef08a' : '#5eead4',
            }}
          >
            {useMockMode ? 'Mode: Mock (Offline)' : 'Mode: Live (Gnani + Gemini)'}
          </button>

          {!useMockMode && (
            <button
              onClick={() => setSelectedBrain(selectedBrain === 'gemini' ? 'ollama' : 'gemini')}
              style={{
                ...styles.toggleBtn,
                background: 'rgba(99, 102, 241, 0.15)',
                borderColor: '#6366f1',
                color: '#c7d2fe',
              }}
              title="Switch conversational brain provider"
            >
              Brain: {selectedBrain.toUpperCase()}
            </button>
          )}
        </div>
      </header>

      {/* Main Experience Area */}
      <main style={styles.main}>
        {/* Presage SmartSpectra Camera Vitals Sensing Card */}
        <SmartSpectraCard
          onCadenceMeasured={setMeasuredCadence}
          recommendedProtocolName={protocol ? formatProtocolName(protocol.protocol) : undefined}
        />

        {/* Real-Time Breathing Session Banner */}
        {isPacingActive && (
          <div style={styles.sessionStatusBanner}>
            <div style={styles.sessionBannerLeft}>
              <Sparkle size={18} color="#2dd4bf" />
              <span>
                Cycle <strong>{currentCycle}</strong> of <strong>{targetCycles}</strong>
              </span>
            </div>
            <div style={styles.sessionBannerRight}>
              <span style={styles.phaseIndicatorBadge}>
                {activePhase === 'begin' ? 'GET READY' : activePhase.replace('_', ' ').toUpperCase()}
              </span>
            </div>
          </div>
        )}

        {/* Central Breathing Orb */}
        <section style={styles.orbSection}>
          <div
            style={{
              ...styles.orbOuter,
              ...(isRecording
                ? styles.orbOuterRecording
                : isProcessing
                ? styles.orbOuterProcessing
                : isPlayingIntroAudio
                ? styles.orbOuterPlaying
                : styles.orbOuterIdle),
              ...getDynamicOrbStyle(),
            }}
            className={
              isPacingActive
                ? ''
                : isRecording
                ? 'orb-recording'
                : isProcessing
                ? 'orb-processing'
                : isPlayingIntroAudio
                ? 'orb-playing'
                : 'orb-idle'
            }
          >
            <div style={styles.orbInner}>
              {isPacingActive ? (
                <div style={{ textAlign: 'center' }}>
                  <div style={styles.pacingPhaseLabel}>
                    {activePhase === 'begin' ? "LET'S BEGIN" : activePhase.replace('_', ' ').toUpperCase()}
                  </div>
                  <div style={styles.pacingCountdown}>{phaseRemaining > 0 ? `${phaseRemaining}s` : '—'}</div>
                  <div style={styles.pacingCycleLabel}>
                    Cycle {currentCycle}/{targetCycles} ({phaseTotalDuration}s)
                  </div>
                </div>
              ) : isRecording ? (
                <div style={{ textAlign: 'center' }}>
                  <Mic size={36} color="#ef4444" />
                  <div style={styles.recordingTimer}>{recordDuration}s</div>
                  <div style={styles.orbSubtext}>Recording...</div>
                </div>
              ) : isProcessing ? (
                <div style={{ textAlign: 'center' }}>
                  <Sparkles size={36} color="#6366f1" />
                  <div style={styles.orbSubtext}>Reasoning...</div>
                </div>
              ) : isPlayingIntroAudio ? (
                <div style={{ textAlign: 'center' }}>
                  <Volume2 size={36} color="#2dd4bf" />
                  <div style={styles.orbSubtext}>Kaveri Voice</div>
                </div>
              ) : activePhase === 'complete' ? (
                <div style={{ textAlign: 'center' }}>
                  <CheckCircle2 size={36} color="#14b8a6" />
                  <div style={styles.orbSubtext}>Completed</div>
                </div>
              ) : (
                <div style={{ textAlign: 'center' }}>
                  <Wind size={36} color="#2dd4bf" />
                  <div style={styles.orbSubtext}>Ready</div>
                </div>
              )}
            </div>
          </div>

          <p style={styles.statusText}>{statusMessage}</p>

          {/* Action Button Controls */}
          <div style={styles.actionRow}>
            {isPacingActive ? (
              // Active Pacing Controls: STOP SESSION
              <button onClick={stopPacingSession} style={styles.stopPacingBtn}>
                <StopCircle size={22} style={{ marginRight: '8px' }} />
                End Breathing Session
              </button>
            ) : !isRecording ? (
              // Idle Controls: RECORD or START BREATHING
              <>
                <button
                  onClick={handleStartRecording}
                  disabled={isProcessing}
                  style={{
                    ...styles.primaryRecordBtn,
                    opacity: isProcessing ? 0.6 : 1,
                  }}
                >
                  <Mic size={22} style={{ marginRight: '8px' }} />
                  Tap to Speak
                </button>

                {protocol && (
                  <button onClick={() => startVoiceGuidedBreathingSession()} style={styles.startPacingBtn}>
                    <PlayCircle size={22} style={{ marginRight: '8px' }} />
                    Start Breathing ({targetCycles} Cycles)
                  </button>
                )}

                {/* Upload File Option */}
                <input
                  type="file"
                  ref={fileInputRef}
                  accept="audio/*,.wav,.webm"
                  style={{ display: 'none' }}
                  onChange={handleFileSelect}
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isRecording || isProcessing}
                  style={styles.uploadBtn}
                  title="Upload pre-recorded audio file (e.g. Recording.wav)"
                >
                  <UploadCloud size={18} style={{ marginRight: '6px' }} />
                  Upload Audio
                </button>
              </>
            ) : (
              // Recording in Progress: STOP & PROCESS
              <button onClick={handleStopRecording} style={styles.stopRecordBtn}>
                <Square size={20} style={{ marginRight: '8px' }} />
                Stop & Process ({recordDuration}s)
              </button>
            )}
          </div>
        </section>

        {/* Error Notice */}
        {errorMessage && (
          <div style={styles.errorBanner}>
            <AlertCircle size={20} color="#f87171" style={{ flexShrink: 0, marginTop: '2px' }} />
            <div>
              <div style={{ fontWeight: 600, color: '#fca5a5' }}>Interaction Notice</div>
              <div style={{ fontSize: '0.9rem', color: '#fecaca' }}>{errorMessage}</div>
            </div>
          </div>
        )}

        {/* Session Results & Guidance Card */}
        {(transcript || assistantText || protocol) && (
          <section style={styles.resultsCard}>
            {/* Transcript */}
            {transcript && (
              <div style={styles.resultItem}>
                <div style={styles.resultLabel}>
                  <Mic size={16} color="#94a3b8" />
                  <span>Transcribed Speech (Gnani Prisma v2.5)</span>
                </div>
                <div style={styles.transcriptBubble}>"{transcript}"</div>
              </div>
            )}

            {/* Assistant Response */}
            {assistantText && (
              <div style={styles.resultItem}>
                <div style={styles.resultLabel}>
                  <Sparkles size={16} color="#2dd4bf" />
                  <span>
                    Assistant Response {isFallback ? '(Deterministic Fallback)' : '(Gemini 3.5 Flash-Lite)'}
                  </span>
                </div>
                <div style={styles.assistantBubble}>{assistantText}</div>
              </div>
            )}

            {/* Intro Audio Player */}
            {introAudioUrl && (
              <div style={styles.resultItem}>
                <div style={styles.resultLabel}>
                  <Volume2 size={16} color="#2dd4bf" />
                  <span>Spoken Guidance (Gnani Timbre v2.5 — Kaveri Persona)</span>
                </div>
                <div style={styles.audioPlayerRow}>
                  <button
                    onClick={() => (isPlayingIntroAudio ? pauseIntroAudio() : playIntroAudio())}
                    style={styles.audioPlayBtn}
                  >
                    {isPlayingIntroAudio ? <Pause size={18} /> : <Play size={18} />}
                    <span>{isPlayingIntroAudio ? 'Pause Voice' : 'Play Voice'}</span>
                  </button>

                  <button onClick={() => playIntroAudio()} style={styles.audioReplayBtn} title="Replay from start">
                    <RotateCcw size={16} />
                    <span>Replay</span>
                  </button>
                </div>
              </div>
            )}

            {/* Recommended Protocol & Guided Breathing Trigger */}
            {protocol && (
              <div style={styles.protocolBox}>
                <div style={styles.protocolHeader}>
                  <Layers size={18} color="#2dd4bf" />
                  <strong>Prescribed Technique: {formatProtocolName(protocol.protocol)}</strong>
                </div>

                {measuredCadence && (
                  <div style={styles.cadenceComparison}>
                    <Activity size={15} color="#2dd4bf" />
                    <span>
                      Detected Respiration: <strong>{measuredCadence} BPM</strong> • Target Protocol Cadence: ~<strong>{Math.round(60 / ((protocol.inhale_s || 4) + (protocol.hold_in_s || 0) + (protocol.exhale_s || 4) + (protocol.hold_out_s || 0)))} BPM</strong>
                    </span>
                  </div>
                )}

                <div style={styles.protocolGrid}>
                  <div style={styles.phasePill}>
                    <span style={styles.pillLabel}>Inhale</span>
                    <strong style={styles.pillValue}>{protocol.inhale_s || 4.0}s</strong>
                  </div>
                  <div style={styles.phasePill}>
                    <span style={styles.pillLabel}>Hold In</span>
                    <strong style={styles.pillValue}>
                      {protocol.hold_in_s ? `${protocol.hold_in_s}s` : 'Skip (0s)'}
                    </strong>
                  </div>
                  <div style={styles.phasePill}>
                    <span style={styles.pillLabel}>Exhale</span>
                    <strong style={styles.pillValue}>{protocol.exhale_s || 4.0}s</strong>
                  </div>
                  <div style={styles.phasePill}>
                    <span style={styles.pillLabel}>Hold Out</span>
                    <strong style={styles.pillValue}>
                      {protocol.hold_out_s ? `${protocol.hold_out_s}s` : 'Skip (0s)'}
                    </strong>
                  </div>
                  <div style={styles.phasePill}>
                    <span style={styles.pillLabel}>Cycles</span>
                    <div style={{ display: 'flex', gap: '4px', justifyContent: 'center' }}>
                      <button
                        onClick={() => setTargetCycles(1)}
                        style={{
                          ...styles.cycleSelectBtn,
                          backgroundColor: targetCycles === 1 ? '#14b8a6' : 'transparent',
                          color: targetCycles === 1 ? '#042f2e' : '#94a3b8',
                        }}
                      >
                        1
                      </button>
                      <button
                        onClick={() => setTargetCycles(protocol.cycles || 4)}
                        style={{
                          ...styles.cycleSelectBtn,
                          backgroundColor: targetCycles === (protocol.cycles || 4) ? '#14b8a6' : 'transparent',
                          color: targetCycles === (protocol.cycles || 4) ? '#042f2e' : '#94a3b8',
                        }}
                      >
                        {protocol.cycles || 4}
                      </button>
                    </div>
                  </div>
                </div>

                {/* Direct Action Trigger */}
                {!isPacingActive && (
                  <div style={{ marginTop: '0.5rem', display: 'flex', justifyContent: 'flex-end' }}>
                    <button
                      onClick={() => startVoiceGuidedBreathingSession()}
                      style={styles.guidedBreathingActionBtn}
                    >
                      <PlayCircle size={18} style={{ marginRight: '6px' }} />
                      Start Voice-Guided Breathing Now
                    </button>
                  </div>
                )}

                <div style={styles.protocolDisclaimer}>
                  <CheckCircle2 size={14} color="#14b8a6" style={{ marginRight: '6px' }} />
                  <span>
                    Deterministic session state machine owned by isolated controller. Non-clinical wellness guidance.
                  </span>
                </div>
              </div>
            )}
          </section>
        )}
      </main>

      {/* Footer */}
      <footer style={styles.footer}>
        <p>BreatheFlow AI — Built for the Gnani AI Internship Challenge & Presage SmartSpectra Challenge</p>
      </footer>
    </div>
  );
}

// Inline styles for zero-dependency modern aesthetics
const styles: { [key: string]: React.CSSProperties } = {
  container: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    position: 'relative',
    padding: '1.5rem',
    maxWidth: '960px',
    margin: '0 auto',
  },
  bgGlow: {
    position: 'fixed',
    top: '-150px',
    left: '50%',
    transform: 'translateX(-50%)',
    width: '600px',
    height: '600px',
    borderRadius: '50%',
    background: 'radial-gradient(circle, rgba(20, 184, 166, 0.12) 0%, rgba(11, 15, 23, 0) 70%)',
    pointerEvents: 'none',
    zIndex: 0,
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingBottom: '1.5rem',
    borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
    position: 'relative',
    zIndex: 1,
    flexWrap: 'wrap',
    gap: '1rem',
  },
  brandRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.85rem',
  },
  logoBadge: {
    width: '42px',
    height: '42px',
    borderRadius: '12px',
    backgroundColor: 'rgba(20, 184, 166, 0.15)',
    border: '1px solid rgba(45, 212, 191, 0.3)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    fontSize: '1.4rem',
    fontWeight: 700,
    color: '#f8fafc',
    letterSpacing: '-0.02em',
  },
  tagline: {
    fontSize: '0.82rem',
    color: '#94a3b8',
  },
  modeControls: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
  },
  badge: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    padding: '0.35rem 0.75rem',
    borderRadius: '20px',
    backgroundColor: 'rgba(255, 255, 255, 0.04)',
    border: '1px solid rgba(255, 255, 255, 0.08)',
  },
  toggleBtn: {
    fontSize: '0.8rem',
    fontWeight: 600,
    padding: '0.4rem 0.85rem',
    borderRadius: '8px',
    borderWidth: '1px',
    borderStyle: 'solid',
    transition: 'all 0.2s ease',
  },
  main: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: '2.5rem 0',
    position: 'relative',
    zIndex: 1,
    width: '100%',
  },
  sessionStatusBanner: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: 'rgba(20, 184, 166, 0.12)',
    border: '1px solid rgba(45, 212, 191, 0.35)',
    borderRadius: '30px',
    padding: '0.5rem 1.25rem',
    width: '100%',
    maxWidth: '460px',
    marginBottom: '1.5rem',
  },
  sessionBannerLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    color: '#99f6e4',
    fontSize: '0.9rem',
  },
  sessionBannerRight: {
    display: 'flex',
    alignItems: 'center',
  },
  phaseIndicatorBadge: {
    backgroundColor: '#14b8a6',
    color: '#042f2e',
    fontWeight: 700,
    fontSize: '0.78rem',
    padding: '0.2rem 0.65rem',
    borderRadius: '12px',
    letterSpacing: '0.04em',
  },
  orbSection: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    marginBottom: '2.5rem',
  },
  orbOuter: {
    width: '210px',
    height: '210px',
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
    cursor: 'pointer',
    marginBottom: '1.5rem',
  },
  orbOuterIdle: {
    background: 'radial-gradient(circle, rgba(20, 184, 166, 0.25) 0%, rgba(15, 23, 42, 0.8) 75%)',
    border: '1px solid rgba(45, 212, 191, 0.4)',
  },
  orbOuterRecording: {
    background: 'radial-gradient(circle, rgba(239, 68, 68, 0.3) 0%, rgba(15, 23, 42, 0.8) 75%)',
    border: '1px solid rgba(248, 113, 113, 0.6)',
  },
  orbOuterProcessing: {
    background: 'radial-gradient(circle, rgba(99, 102, 241, 0.35) 0%, rgba(15, 23, 42, 0.8) 75%)',
    border: '1px solid rgba(129, 140, 248, 0.5)',
  },
  orbOuterPlaying: {
    background: 'radial-gradient(circle, rgba(20, 184, 166, 0.4) 0%, rgba(15, 23, 42, 0.8) 75%)',
    border: '1px solid rgba(94, 234, 212, 0.7)',
  },
  orbInner: {
    width: '150px',
    height: '150px',
    borderRadius: '50%',
    backgroundColor: '#0d131f',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    border: '1px solid rgba(255, 255, 255, 0.08)',
  },
  recordingTimer: {
    fontSize: '1.3rem',
    fontWeight: 700,
    color: '#f87171',
    marginTop: '4px',
  },
  orbSubtext: {
    fontSize: '0.78rem',
    color: '#94a3b8',
    letterSpacing: '0.05em',
    textTransform: 'uppercase',
    marginTop: '4px',
  },
  pacingPhaseLabel: {
    fontSize: '0.9rem',
    fontWeight: 700,
    color: '#2dd4bf',
    letterSpacing: '0.06em',
  },
  pacingCountdown: {
    fontSize: '2.2rem',
    fontWeight: 800,
    color: '#f8fafc',
    lineHeight: 1.1,
    marginTop: '2px',
  },
  pacingCycleLabel: {
    fontSize: '0.72rem',
    color: '#94a3b8',
    textTransform: 'uppercase',
    marginTop: '3px',
  },
  statusText: {
    fontSize: '0.95rem',
    color: '#94a3b8',
    marginBottom: '1.5rem',
    textAlign: 'center',
    maxWidth: '480px',
  },
  actionRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.85rem',
    flexWrap: 'wrap',
    justifyContent: 'center',
  },
  primaryRecordBtn: {
    display: 'flex',
    alignItems: 'center',
    backgroundColor: '#14b8a6',
    color: '#042f2e',
    fontWeight: 600,
    padding: '0.85rem 1.75rem',
    borderRadius: '30px',
    fontSize: '1rem',
    boxShadow: '0 4px 20px rgba(20, 184, 166, 0.35)',
    transition: 'transform 0.15s ease, background 0.15s ease',
  },
  startPacingBtn: {
    display: 'flex',
    alignItems: 'center',
    backgroundColor: 'rgba(20, 184, 166, 0.2)',
    border: '1px solid #14b8a6',
    color: '#5eead4',
    fontWeight: 600,
    padding: '0.85rem 1.5rem',
    borderRadius: '30px',
    fontSize: '1rem',
  },
  stopPacingBtn: {
    display: 'flex',
    alignItems: 'center',
    backgroundColor: '#dc2626',
    color: '#ffffff',
    fontWeight: 700,
    padding: '0.85rem 2rem',
    borderRadius: '30px',
    fontSize: '1.05rem',
    boxShadow: '0 4px 25px rgba(220, 38, 38, 0.45)',
  },
  stopRecordBtn: {
    display: 'flex',
    alignItems: 'center',
    backgroundColor: '#ef4444',
    color: '#ffffff',
    fontWeight: 600,
    padding: '0.85rem 1.75rem',
    borderRadius: '30px',
    fontSize: '1rem',
    boxShadow: '0 4px 20px rgba(239, 68, 68, 0.4)',
  },
  uploadBtn: {
    display: 'flex',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    color: '#94a3b8',
    border: '1px solid rgba(255, 255, 255, 0.12)',
    fontWeight: 500,
    padding: '0.85rem 1.25rem',
    borderRadius: '30px',
    fontSize: '0.9rem',
  },
  errorBanner: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: '12px',
    backgroundColor: 'rgba(239, 68, 68, 0.12)',
    border: '1px solid rgba(248, 113, 113, 0.3)',
    borderRadius: '12px',
    padding: '1rem',
    width: '100%',
    maxWidth: '680px',
    marginBottom: '1.5rem',
  },
  resultsCard: {
    width: '100%',
    maxWidth: '680px',
    backgroundColor: '#111827',
    borderRadius: '16px',
    border: '1px solid rgba(255, 255, 255, 0.08)',
    padding: '1.5rem',
    display: 'flex',
    flexDirection: 'column',
    gap: '1.25rem',
    boxShadow: '0 10px 30px rgba(0, 0, 0, 0.4)',
  },
  resultItem: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.45rem',
  },
  resultLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    fontSize: '0.8rem',
    fontWeight: 600,
    color: '#94a3b8',
    textTransform: 'uppercase',
    letterSpacing: '0.04em',
  },
  transcriptBubble: {
    backgroundColor: 'rgba(255, 255, 255, 0.04)',
    border: '1px solid rgba(255, 255, 255, 0.06)',
    borderRadius: '10px',
    padding: '0.85rem 1rem',
    fontSize: '0.95rem',
    color: '#cbd5e1',
    fontStyle: 'italic',
  },
  assistantBubble: {
    backgroundColor: 'rgba(20, 184, 166, 0.08)',
    border: '1px solid rgba(45, 212, 191, 0.25)',
    borderRadius: '10px',
    padding: '0.95rem 1.1rem',
    fontSize: '1rem',
    color: '#f0fdfa',
    lineHeight: 1.6,
  },
  audioPlayerRow: {
    display: 'flex',
    gap: '0.75rem',
    marginTop: '0.25rem',
  },
  audioPlayBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    backgroundColor: '#14b8a6',
    color: '#042f2e',
    fontWeight: 600,
    padding: '0.6rem 1.15rem',
    borderRadius: '20px',
    fontSize: '0.88rem',
  },
  audioReplayBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    backgroundColor: 'rgba(255, 255, 255, 0.06)',
    color: '#cbd5e1',
    border: '1px solid rgba(255, 255, 255, 0.1)',
    fontWeight: 500,
    padding: '0.6rem 1rem',
    borderRadius: '20px',
    fontSize: '0.85rem',
  },
  protocolBox: {
    marginTop: '0.5rem',
    backgroundColor: 'rgba(15, 23, 42, 0.65)',
    border: '1px solid rgba(45, 212, 191, 0.25)',
    borderRadius: '12px',
    padding: '1.25rem',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.85rem',
  },
  protocolHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    color: '#5eead4',
    fontSize: '0.95rem',
  },
  cadenceComparison: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    fontSize: '0.8rem',
    color: '#5eead4',
    backgroundColor: 'rgba(20, 184, 166, 0.1)',
    border: '1px solid rgba(20, 184, 166, 0.25)',
    padding: '0.35rem 0.75rem',
    borderRadius: '8px',
    margin: '0.4rem 0',
  },
  protocolGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(80px, 1fr))',
    gap: '0.6rem',
  },
  phasePill: {
    backgroundColor: 'rgba(255, 255, 255, 0.03)',
    border: '1px solid rgba(255, 255, 255, 0.06)',
    borderRadius: '8px',
    padding: '0.6rem',
    textAlign: 'center',
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
  },
  pillLabel: {
    fontSize: '0.72rem',
    color: '#94a3b8',
    textTransform: 'uppercase',
  },
  pillValue: {
    fontSize: '1rem',
    color: '#f8fafc',
  },
  cycleSelectBtn: {
    padding: '2px 8px',
    borderRadius: '6px',
    border: '1px solid rgba(255, 255, 255, 0.1)',
    fontSize: '0.85rem',
    fontWeight: 600,
    cursor: 'pointer',
  },
  guidedBreathingActionBtn: {
    display: 'flex',
    alignItems: 'center',
    backgroundColor: '#14b8a6',
    color: '#042f2e',
    fontWeight: 600,
    padding: '0.6rem 1.2rem',
    borderRadius: '20px',
    fontSize: '0.9rem',
    cursor: 'pointer',
    border: 'none',
  },
  protocolDisclaimer: {
    display: 'flex',
    alignItems: 'center',
    fontSize: '0.76rem',
    color: '#94a3b8',
    marginTop: '0.25rem',
  },
  footer: {
    textAlign: 'center',
    paddingTop: '2rem',
    borderTop: '1px solid rgba(255, 255, 255, 0.06)',
    color: '#64748b',
    fontSize: '0.8rem',
    position: 'relative',
    zIndex: 1,
  },
};

export default App;
