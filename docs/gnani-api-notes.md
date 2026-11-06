# Gnani API Technical Reference & Integration Notes

## 1. Overview & Official Documentation Sources

This document details the official API specifications for **Gnani Prisma v2.5** (Speech-to-Text) and **Gnani Timbre v2.5** (Text-to-Speech) hosted under Gnani's **Vachana AI** developer platform.

### Official Documentation & Resources Consulted
- **Gnani AI Developer Platform**: [`https://api.vachana.ai`](https://api.vachana.ai) / [`https://gnani.ai`](https://gnani.ai)
- **STT (Prisma) REST API Reference**: `POST /stt/v3` ([Gnani STT Docs](https://gnani.ai))
- **TTS (Timbre) REST API Reference**: `POST /api/v1/tts/inference` ([Gnani TTS Docs](https://gnani.ai))
- **Official Python Client**: `gnani-vachana` package ([PyPI gnani-vachana](https://pypi.org/project/gnani-vachana/))

---

## 2. Authentication & Environment Conventions

### Authentication Protocol
- **Header Key**: `X-API-Key-ID`
- **Value**: The raw API key token issued by the Gnani Developer Portal.
- **Example Header**:
  ```http
  X-API-Key-ID: your_gnani_api_key_here
  ```

### Environment Variable Convention
To avoid hardcoding secrets, all components must read from standard environment variables:
- `GNANI_API_KEY`: Primary API key for both Prisma and Timbre calls.
- `GNANI_BASE_URL`: Base URL override (default: `https://api.vachana.ai`).

---

## 3. Speech-to-Text (STT): Gnani Prisma v2.5

### Endpoint Specifications
- **HTTP Method & Path**: `POST https://api.vachana.ai/stt/v3`
- **Content-Type**: `multipart/form-data`
- **Headers**:
  - `X-API-Key-ID: ${GNANI_API_KEY}`

### Request Form Parameters

| Parameter | Type | Required | Description | Verified Value |
|---|---|---|---|---|
| `audio_file` | File (Binary) | **Yes** | Audio recording buffer or file pointer (WAV / MP3 / PCM). Recommended $\le 30$ seconds for synchronous REST. | Validated in official cURL spec. |
| `language_code` | String | **Yes** | Target language tag (e.g. `en-IN` for Indian English, `hi-IN` for Hindi). | Validated. |
| `model` | String | No (Default) | STT model identifier. Default is `prisma-v2.5` (or `prisma-v3.0`). | Validated. |
| `format` | String | No | Formatting style: `transcribe` (clean formatting for voice bots), `verbatim`, `codemix`. | Validated. |

### Expected Response Payload (HTTP 200)
```json
{
  "success": true,
  "request_id": "5c1f0a52-6d62-4e0b-9a38-2f2b8a8f7a10",
  "transcript": "I want to start a box breathing session",
  "model": "prisma-v2.5",
  "detected_language_code": "en-IN",
  "processing_time": 0.62,
  "end_to_end_latency": 0.71
}
```

---

## 4. Text-to-Speech (TTS): Gnani Timbre v2.5

### Endpoint Specifications
- **HTTP Method & Path**: `POST https://api.vachana.ai/api/v1/tts/inference`
- **Content-Type**: `application/json`
- **Headers**:
  - `Content-Type: application/json`
  - `X-API-Key-ID: ${GNANI_API_KEY}`

### JSON Request Body Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `text` | String | **Yes** | The plain text string to synthesize into speech. |
| `model` | String | No | Target model: `"timbre-v2.5"` (or `"timbre-v3.0"`). |
| `language` | String | No | Language code (e.g., `"en-IN"`, `"hi-IN"`). |
| `voice` | String | No | Voice persona name (e.g., `"Kaveri"` for English, `"Nalini"` for Hindi). |
| `speed` | Float | No | Speaking pace modifier (default: `1.0`; for calming guidance, `0.9`–`0.95` is ideal). |
| `audio_config` | Object | No | Output format configuration: `sample_rate` (16000 / 24000 / 48000), `encoding` (`"linear_pcm"`), `container` (`"wav"`). |

### Request Example
```json
{
  "text": "Welcome to BreatheFlow. Get comfortable, and let's begin with a gentle breath in.",
  "model": "timbre-v2.5",
  "language": "en-IN",
  "voice": "Kaveri",
  "speed": 0.95,
  "audio_config": {
    "sample_rate": 24000,
    "num_channels": 1,
    "encoding": "linear_pcm",
    "container": "wav"
  }
}
```

### Response Payload
- **HTTP Status**: `200 OK`
- **Content-Type**: `audio/wav` (or binary stream container specified)
- **Body**: Binary audio bytes ready for immediate playback or saving to disk.

---

## 5. Architectural Assessment: Is REST Suitable for the Initial MVP?

### Verdict: **YES — REST is optimal for the M1 MVP**

1. **Simplicity & Deadline Alignment**: The Gnani challenge deadline is **November 10, 2026, 11:59 PM IST** (~3 days away). REST drastically reduces implementation complexity and state management risk compared to full-duplex WebSockets.
2. **Turn-Based Interaction Nature**: BreatheFlow AI interaction is naturally turn-based:
   - User speaks a phrase $\rightarrow$ REST STT transcribes ($\sim 600\text{ms}$).
   - Qwen3-4B generates short guidance ($\sim 400\text{ms}$).
   - REST TTS synthesizes speech response ($\sim 500\text{ms}$).
   - **Total turn turnaround: $\sim 1.5\text{s}$**, which is well within acceptable voice assistant conversational thresholds.
3. **Decoupled Breathing Execution**: Once the session begins, pacing cues are handled **locally and deterministically** by the breathing session controller. The system does not depend on cloud streaming round-trips for each second of inhalation/exhalation.
4. **Clean Upgrade Path**: By encapsulating REST calls behind abstract interfaces (`STTProvider`, `TTSProvider`), streaming WebSockets (`/stt/v3/stream` and `/api/v1/tts/sse`) can be added in subsequent milestones without rewriting orchestrator logic.

---

## 6. Verified Details vs. Assumptions & Open Questions

| Topic | Verified Detail | Assumption / Open Question Requiring Key |
|---|---|---|
| **Base URL** | `https://api.vachana.ai` is the verified platform host. | Assumed active and reachable from local development environment without firewall blocks. |
| **Authentication** | `X-API-Key-ID` header is confirmed in official cURL docs. | Need to verify rate limits and quota bounds for challenge developer keys. |
| **STT Response** | Field is named `transcript` (not `text`). | Need to verify whether `confidence` score is returned per utterance in `prisma-v2.5`. |
| **TTS Voices** | `"Kaveri"` and `"Nalini"` are valid personas. | Need to test which voice persona sounds most natural and soothing for English mindfulness pacing. |
| **Model Names** | `model: "prisma-v2.5"` and `model: "timbre-v2.5"` are accepted. | Need to verify if omitting `model` defaults to `v2.5` or `v3.0`. |
