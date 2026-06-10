# Submission

## Overview

Real-time English to Hindi voice translator.

## Architecture

```text
Microphone
↓
Streaming ASR
↓
Transcript Stabilizer
↓
Groq Translation
↓
Hindi TTS
↓
Audio Output
```

## Setup

```bash
cd submission
export GROQ_API_KEY="your-groq-api-key"
pip install -r requirements.txt
```

## Running Instructions

Terminal 1:

```bash
python asr_server.py
```

Terminal 2:

```bash
python app.py
```

## Design Decisions

* WhisperLive-style streaming ASR is implemented locally to keep the submission self-contained.
* Groq is used for low-latency English-to-Hindi translation.
* Edge-TTS is used for natural Hindi speech synthesis.
* Transcript stabilization avoids translating partial ASR hypotheses too early.
* TTS feedback suppression avoids self-transcription loops.

## Edge Cases Handled

* Duplicate transcripts
* Partial transcript stabilization
* Named entity preservation
* Filler removal
* TTS feedback suppression

## Limitations

* ASR accuracy depends on microphone quality.
* Strong accents may affect transcription.
* Translation quality depends on LLM output.
* Network latency affects translation response time.
