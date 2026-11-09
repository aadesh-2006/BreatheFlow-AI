import React, { useState, useEffect, useRef } from 'react';
import {
  Camera,
  CameraOff,
  Activity,
  Heart,
  Wind,
  ShieldCheck,
  AlertCircle,
  HelpCircle,
  CheckCircle,
  Monitor,
} from 'lucide-react';
import {
  smartSpectraController,
  type VitalsReading,
} from '../services/smartSpectra';

interface SmartSpectraCardProps {
  onCadenceMeasured?: (cadenceBpm: number) => void;
  recommendedProtocolName?: string;
}

export const SmartSpectraCard: React.FC<SmartSpectraCardProps> = ({
  onCadenceMeasured,
  recommendedProtocolName,
}) => {
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isSupported, setIsSupported] = useState(false);
  const [isConfigured, setIsConfigured] = useState<boolean | null>(null);
  const [configNotice, setConfigNotice] = useState<string | null>(null);
  const [statusText, setStatusText] = useState('Idle');
  const [vitals, setVitals] = useState<VitalsReading>({
    breathingRateBpm: null,
    breathingConfidence: null,
    breathingStable: false,
    pulseRateBpm: null,
    pulseConfidence: null,
    pulseStable: false,
    validationStatus: 'Idle',
    validationCode: -1,
    processingStatus: 'Idle',
    timestampUs: 0,
  });

  const videoRef = useRef<HTMLVideoElement | null>(null);

  // Check Electron & SmartSpectra environment on mount
  useEffect(() => {
    const isBridgeAvailable = smartSpectraController.isAvailable();
    setIsSupported(isBridgeAvailable);

    // Query developer configuration status securely from Electron main process
    const win = window as unknown as {
      electronAPI?: {
        isElectron?: boolean;
        checkSmartSpectraConfig?: () => Promise<{ configured: boolean }>;
      };
    };

    if (win.electronAPI?.checkSmartSpectraConfig) {
      win.electronAPI.checkSmartSpectraConfig().then((res) => {
        setIsConfigured(res.configured);
        if (!res.configured) {
          setConfigNotice(
            'Developer Configuration Notice: SMARTSPECTRA_API_KEY is not configured in the root .env file. Add your developer key to .env to enable camera vitals sensing.'
          );
        } else {
          setConfigNotice(null);
        }
      });
    }

    // Subscribe to vitals updates
    const unsubVitals = smartSpectraController.subscribeVitals((v) => {
      setVitals(v);
      if (v.breathingRateBpm && onCadenceMeasured) {
        onCadenceMeasured(v.breathingRateBpm);
      }
    });

    // Subscribe to status updates
    const unsubStatus = smartSpectraController.subscribeStatus((st) => {
      setStatusText(st);
    });

    // Subscribe to camera stream
    const unsubStream = smartSpectraController.subscribeStream((stream) => {
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play().catch((err) => {
          console.warn('Video preview playback error:', err);
        });
      }
    });

    return () => {
      unsubVitals();
      unsubStatus();
      unsubStream();
      smartSpectraController.stop();
    };
  }, [onCadenceMeasured]);

  const handleToggleCamera = async () => {
    if (isCameraActive) {
      await smartSpectraController.stop();
      setIsCameraActive(false);
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    } else {
      if (isConfigured === false) {
        setStatusText(
          'Developer Configuration Required: SMARTSPECTRA_API_KEY is missing from root .env. Please configure your key in .env.'
        );
        return;
      }

      try {
        await smartSpectraController.start();
        setIsCameraActive(true);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        if (msg.includes('authentication failed') || msg.includes('OAuth')) {
          setStatusText(
            'SmartSpectra authentication failed: The developer SMARTSPECTRA_API_KEY was rejected by Presage authorization servers. Please verify the key in .env.'
          );
        } else {
          setStatusText(`Failed to activate camera: ${msg}`);
        }
      }
    }
  };

  return (
    <div style={cardStyles.container}>
      {/* Header */}
      <div style={cardStyles.headerRow}>
        <div style={cardStyles.titleGroup}>
          <div style={cardStyles.iconBadge}>
            <Activity size={18} color="#2dd4bf" />
          </div>
          <div>
            <div style={cardStyles.title}>Presage SmartSpectra™ Vitals Sensing</div>
            <div style={cardStyles.subtitle}>
              Contactless camera respiration & pulse metrics powered by @smartspectra/node-sdk
            </div>
          </div>
        </div>

        {/* Runtime Badge */}
        <div style={cardStyles.runtimeBadge}>
          {isSupported ? (
            <span style={{ color: '#2dd4bf', display: 'flex', alignItems: 'center', gap: '4px' }}>
              <ShieldCheck size={14} /> Electron Desktop Native
            </span>
          ) : (
            <span style={{ color: '#eab308', display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Monitor size={14} /> Browser Web Mode
            </span>
          )}
        </div>
      </div>

      {/* Developer Configuration Notice (Shown only if developer has not set .env) */}
      {configNotice && (
        <div style={cardStyles.devNoticeBanner}>
          <AlertCircle size={15} color="#f59e0b" style={{ flexShrink: 0, marginTop: '2px' }} />
          <span>{configNotice}</span>
        </div>
      )}

      {/* Main Content Grid */}
      <div style={cardStyles.grid}>
        {/* Left: Camera Preview Viewport */}
        <div style={cardStyles.previewContainer}>
          <div style={cardStyles.viewport}>
            <video
              ref={videoRef}
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'cover',
                display: isCameraActive ? 'block' : 'none',
              }}
              playsInline
              muted
            />

            {!isCameraActive && (
              <div style={cardStyles.placeholder}>
                <Camera size={36} color="#64748b" />
                <span style={cardStyles.placeholderText}>
                  Camera inactive. Click "Enable Camera Vitals" to grant permission and begin sensing.
                </span>
              </div>
            )}

            {/* Target Reticle Overlay */}
            {isCameraActive && (
              <div style={cardStyles.reticleOverlay}>
                <div style={cardStyles.faceBox} />
                <div style={cardStyles.overlayStatus}>
                  {vitals.validationStatus}
                </div>
              </div>
            )}
          </div>

          {/* Consent / Controls Action Row */}
          <div style={cardStyles.controlsRow}>
            {isSupported ? (
              <button
                onClick={handleToggleCamera}
                style={{
                  ...cardStyles.toggleBtn,
                  backgroundColor: isCameraActive ? '#dc2626' : '#0d9488',
                  color: '#ffffff',
                }}
              >
                {isCameraActive ? (
                  <>
                    <CameraOff size={16} style={{ marginRight: '6px' }} /> Stop Camera Vitals
                  </>
                ) : (
                  <>
                    <Camera size={16} style={{ marginRight: '6px' }} /> Enable Camera Vitals
                  </>
                )}
              </button>
            ) : (
              <div style={cardStyles.browserNotice}>
                <Monitor size={16} color="#eab308" style={{ marginRight: '6px' }} />
                <span>Run <code>npm run electron:dev</code> to activate native SmartSpectra camera vitals</span>
              </div>
            )}
          </div>
        </div>

        {/* Right: Metrics & Physiological Adaptation */}
        <div style={cardStyles.metricsContainer}>
          {/* Respiration Metric Card */}
          <div style={cardStyles.metricCard}>
            <div style={cardStyles.metricCardHeader}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Wind size={16} color="#2dd4bf" />
                <span style={cardStyles.metricTitle}>Respiration Rate (Breaths/Min)</span>
              </div>
              {vitals.breathingStable && (
                <span style={cardStyles.stableBadge}>
                  <CheckCircle size={12} /> Stable
                </span>
              )}
            </div>

            <div style={cardStyles.metricValueRow}>
              <div style={cardStyles.metricLargeNumber}>
                {vitals.breathingRateBpm !== null ? vitals.breathingRateBpm : '—'}
              </div>
              <div style={cardStyles.metricUnit}>BPM</div>
              {vitals.breathingConfidence !== null && (
                <div style={cardStyles.confidenceText}>
                  Confidence: {vitals.breathingConfidence}%
                </div>
              )}
            </div>

            <div style={cardStyles.metricFooter}>
              {vitals.breathingRateBpm ? (
                vitals.breathingRateBpm > 16 ? (
                  <span style={{ color: '#f59e0b' }}>
                    Rapid baseline detected ({vitals.breathingRateBpm} BPM). Calming protocol suggested to reduce sympathetic load.
                  </span>
                ) : (
                  <span style={{ color: '#2dd4bf' }}>
                    Optimal relaxed baseline ({vitals.breathingRateBpm} BPM).
                  </span>
                )
              ) : (
                <span style={{ color: '#64748b' }}>Awaiting initial breathing cycle analysis (~15-30s)...</span>
              )}
            </div>
          </div>

          {/* Pulse / Heart Rate Metric Card */}
          <div style={cardStyles.metricCard}>
            <div style={cardStyles.metricCardHeader}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Heart size={16} color="#f43f5e" />
                <span style={cardStyles.metricTitle}>Pulse Rate (Heart Rate)</span>
              </div>
              {vitals.pulseStable && (
                <span style={cardStyles.stableBadge}>
                  <CheckCircle size={12} /> Stable
                </span>
              )}
            </div>

            <div style={cardStyles.metricValueRow}>
              <div style={cardStyles.metricLargeNumber}>
                {vitals.pulseRateBpm !== null ? vitals.pulseRateBpm : '—'}
              </div>
              <div style={cardStyles.metricUnit}>BPM</div>
              {vitals.pulseConfidence !== null && (
                <div style={cardStyles.confidenceText}>
                  Confidence: {vitals.pulseConfidence}%
                </div>
              )}
            </div>

            <div style={cardStyles.metricFooter}>
              <span style={{ color: '#94a3b8' }}>
                Status: {vitals.processingStatus} • {statusText}
              </span>
            </div>
          </div>

          {/* Guided Protocol Integration Callout */}
          <div style={cardStyles.adaptationCallout}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
              <HelpCircle size={15} color="#818cf8" />
              <strong style={{ fontSize: '0.85rem', color: '#c7d2fe' }}>
                Adaptive Protocol Alignment
              </strong>
            </div>
            <p style={cardStyles.calloutText}>
              {vitals.breathingRateBpm ? (
                <>
                  Camera-measured baseline: <strong>{vitals.breathingRateBpm} BPM</strong>.
                  {recommendedProtocolName ? (
                    <> Selected protocol <strong>{recommendedProtocolName}</strong> will guide your breath down to a restorative cadence.</>
                  ) : (
                    <> Use your voice above to let Gemini select a matching protocol based on your baseline.</>
                  )}
                </>
              ) : (
                'Center your upper chest and face in view. Presage SmartSpectra will establish your baseline respiration before and during guided breathing.'
              )}
            </p>
          </div>
        </div>
      </div>

      {/* Non-Clinical Disclaimer */}
      <div style={cardStyles.disclaimerRow}>
        <AlertCircle size={13} color="#94a3b8" style={{ flexShrink: 0, marginTop: '1px' }} />
        <span>
          <strong>Wellness Biofeedback Notice:</strong> Camera-derived metrics from Presage SmartSpectra™ are for general wellness, mindfulness, and breathing feedback. They do not constitute medical diagnosis, monitoring, or clinical care.
        </span>
      </div>
    </div>
  );
};

const cardStyles: { [key: string]: React.CSSProperties } = {
  container: {
    width: '100%',
    backgroundColor: 'rgba(15, 23, 42, 0.65)',
    border: '1px solid rgba(45, 212, 191, 0.25)',
    borderRadius: '16px',
    padding: '1.25rem',
    marginBottom: '2rem',
    backdropFilter: 'blur(12px)',
    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.35)',
  },
  headerRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    flexWrap: 'wrap',
    gap: '0.75rem',
    marginBottom: '1rem',
    paddingBottom: '0.75rem',
    borderBottom: '1px solid rgba(255, 255, 255, 0.06)',
  },
  titleGroup: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
  },
  iconBadge: {
    width: '36px',
    height: '36px',
    borderRadius: '10px',
    backgroundColor: 'rgba(20, 184, 166, 0.15)',
    border: '1px solid rgba(45, 212, 191, 0.3)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    fontSize: '1.05rem',
    fontWeight: 700,
    color: '#f8fafc',
    letterSpacing: '-0.01em',
  },
  subtitle: {
    fontSize: '0.78rem',
    color: '#94a3b8',
  },
  runtimeBadge: {
    fontSize: '0.75rem',
    fontWeight: 600,
    backgroundColor: 'rgba(255, 255, 255, 0.04)',
    border: '1px solid rgba(255, 255, 255, 0.08)',
    padding: '0.3rem 0.65rem',
    borderRadius: '20px',
  },
  devNoticeBanner: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: '8px',
    fontSize: '0.78rem',
    color: '#fef08a',
    backgroundColor: 'rgba(234, 179, 8, 0.12)',
    border: '1px solid rgba(234, 179, 8, 0.3)',
    borderRadius: '8px',
    padding: '0.6rem 0.85rem',
    marginBottom: '1rem',
    lineHeight: 1.4,
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
    gap: '1.25rem',
  },
  previewContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.75rem',
  },
  viewport: {
    position: 'relative',
    width: '100%',
    height: '210px',
    borderRadius: '12px',
    backgroundColor: '#0a0d14',
    border: '1px solid rgba(255, 255, 255, 0.08)',
    overflow: 'hidden',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  placeholder: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    textAlign: 'center',
    gap: '8px',
    padding: '1rem',
  },
  placeholderText: {
    fontSize: '0.8rem',
    color: '#64748b',
    maxWidth: '220px',
  },
  reticleOverlay: {
    position: 'absolute',
    inset: 0,
    pointerEvents: 'none',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '12px',
  },
  faceBox: {
    width: '110px',
    height: '130px',
    border: '2px dashed rgba(45, 212, 191, 0.6)',
    borderRadius: '16px',
    marginTop: '6px',
  },
  overlayStatus: {
    fontSize: '0.75rem',
    backgroundColor: 'rgba(15, 23, 42, 0.85)',
    color: '#5eead4',
    padding: '0.2rem 0.6rem',
    borderRadius: '8px',
    border: '1px solid rgba(45, 212, 191, 0.3)',
  },
  controlsRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.65rem',
    flexWrap: 'wrap',
  },
  toggleBtn: {
    display: 'flex',
    alignItems: 'center',
    fontWeight: 600,
    fontSize: '0.85rem',
    padding: '0.5rem 1.25rem',
    borderRadius: '8px',
    border: 'none',
    cursor: 'pointer',
    transition: 'opacity 0.15s ease',
  },
  browserNotice: {
    display: 'flex',
    alignItems: 'center',
    fontSize: '0.78rem',
    color: '#cbd5e1',
    backgroundColor: 'rgba(234, 179, 8, 0.1)',
    border: '1px solid rgba(234, 179, 8, 0.25)',
    padding: '0.45rem 0.75rem',
    borderRadius: '8px',
  },
  metricsContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.75rem',
  },
  metricCard: {
    backgroundColor: 'rgba(11, 15, 25, 0.7)',
    border: '1px solid rgba(255, 255, 255, 0.07)',
    borderRadius: '12px',
    padding: '0.85rem',
  },
  metricCardHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '0.4rem',
  },
  metricTitle: {
    fontSize: '0.78rem',
    fontWeight: 600,
    color: '#cbd5e1',
  },
  stableBadge: {
    fontSize: '0.7rem',
    fontWeight: 600,
    color: '#34d399',
    backgroundColor: 'rgba(16, 185, 129, 0.12)',
    padding: '0.15rem 0.45rem',
    borderRadius: '6px',
    display: 'flex',
    alignItems: 'center',
    gap: '3px',
  },
  metricValueRow: {
    display: 'flex',
    alignItems: 'baseline',
    gap: '6px',
  },
  metricLargeNumber: {
    fontSize: '1.9rem',
    fontWeight: 800,
    color: '#f8fafc',
    lineHeight: 1,
  },
  metricUnit: {
    fontSize: '0.85rem',
    fontWeight: 600,
    color: '#94a3b8',
  },
  confidenceText: {
    fontSize: '0.72rem',
    color: '#64748b',
    marginLeft: 'auto',
  },
  metricFooter: {
    fontSize: '0.74rem',
    marginTop: '0.35rem',
  },
  adaptationCallout: {
    backgroundColor: 'rgba(99, 102, 241, 0.1)',
    border: '1px solid rgba(99, 102, 241, 0.25)',
    borderRadius: '10px',
    padding: '0.75rem',
  },
  calloutText: {
    fontSize: '0.78rem',
    color: '#cbd5e1',
    lineHeight: 1.4,
  },
  disclaimerRow: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: '6px',
    marginTop: '1rem',
    paddingTop: '0.75rem',
    borderTop: '1px solid rgba(255, 255, 255, 0.05)',
    fontSize: '0.7rem',
    color: '#64748b',
    lineHeight: 1.35,
  },
};
