#!/usr/bin/env python3
"""
OpenAI-Compatible TTS Server with Piper + Autotune + RVC (Simplified)
Added autotune layer: pitch up one octave + C major scale quantization

Pipeline:
1. Command-line Piper TTS
2. Autotune processing (pitch up + C scale)
3. RVC CPU conversion
4. FFmpeg format conversion
"""

import os
import tempfile
import asyncio
import logging
from typing import Optional, Literal
from pathlib import Path
import torch
import numpy as np

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fix PyTorch weights_only loading issue
try:
    import fairseq.data.dictionary

    torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])
    logger.info("Added fairseq.data.dictionary.Dictionary to safe globals")
except Exception as e:
    logger.warning(f"Could not add fairseq safe globals: {e}")

app = FastAPI(
    title="OpenAI-Compatible TTS with Piper + Autotune + RVC",
    description="TTS server with autotune layer (pitch up + C scale quantization)",
    version="3.1.0",
)


class SpeechRequest(BaseModel):
    model: str = Field(default="tts-1")
    input: str = Field(..., max_length=4096)
    voice: str = Field(default="alloy")
    response_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]] = Field(default="mp3")
    speed: Optional[float] = Field(default=1.0, ge=0.25, le=4.0)


class TTSServer:
    def __init__(self):
        self.piper_model_path = None
        self.rvc_instance = None
        self.temp_dir = tempfile.mkdtemp()

        # E major scale frequencies (E4 and up, scaled up one octave from E4)
        self.c_scale_freqs = [
            329.63,  # E4
            369.99,  # F#4
            415.30,  # G#4
            440.00,  # A4
            493.88,  # B4
            554.37,  # C#5
            622.25,  # D#5
            659.25,  # E5
            739.99,  # F#5
            830.61,  # G#5
            880.00,  # A5
            987.77,  # B5
            1108.73,  # C#6
            1244.51,  # D#6
            1318.51,  # E6
        ]

        self.setup_models()

    def setup_models(self):
        """Setup models - simplified"""
        self.setup_piper_model()
        self.setup_rvc_model()

    def setup_piper_model(self):
        """Setup Piper TTS model"""
        model_dir = Path("models/piper")
        model_dir.mkdir(parents=True, exist_ok=True)

        model_name = "en_US-hfc_female-medium"
        onnx_path = model_dir / f"{model_name}.onnx"
        json_path = model_dir / f"{model_name}.onnx.json"

        if not onnx_path.exists() or not json_path.exists():
            logger.info("Downloading Piper TTS model...")
            import urllib.request

            base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium"

            if not onnx_path.exists():
                urllib.request.urlretrieve(f"{base_url}/en_US-amy-medium.onnx?download=true", onnx_path)
            if not json_path.exists():
                urllib.request.urlretrieve(f"{base_url}/en_US-amy-medium.onnx.json?download=true", json_path)

        # Verify model files
        if onnx_path.stat().st_size > 0:
            logger.info(f"Piper model files verified: {onnx_path} ({onnx_path.stat().st_size} bytes)")

        self.piper_model_path = str(onnx_path)

    def setup_rvc_model(self):
        """Setup RVC model - CPU only"""
        model_dir = Path("models/rvc")
        model_dir.mkdir(parents=True, exist_ok=True)

        from rvc_python.infer import RVCInference

        model_path = model_dir / "hatsune_miku.pth"
        if not model_path.exists():
            logger.info("Downloading Hatsune Miku RVC model...")
            import urllib.request

            urllib.request.urlretrieve(
                "https://huggingface.co/voice-models/Hatsune_Miku/resolve/main/Hatsune_Miku.pth", model_path
            )

        # Force CPU mode
        logger.info("FORCING RVC to use CPU mode (device: cpu)")
        self.rvc_instance = RVCInference(device="cpu")
        self.rvc_instance.load_model(str(model_path))

        logger.info("rvc-python initialized successfully on cpu (CPU forced)")
        logger.info("RVC setup successful with rvc_python (CPU mode)")

    async def generate_with_piper(self, text: str, output_path: str):
        """Generate audio using command-line Piper TTS"""
        logger.info(f"Generating speech with Piper TTS: '{text[:50]}...' -> {output_path}")

        cmd = ["piper", "--model", self.piper_model_path, "--output-file", output_path]

        process = await asyncio.create_subprocess_exec(
            *cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate(input=text.encode())

        if process.returncode == 0 and Path(output_path).exists():
            file_size = Path(output_path).stat().st_size
            logger.info(f"Command-line Piper TTS successful: {output_path} ({file_size} bytes)")
        else:
            raise Exception(f"Piper TTS failed: {stderr.decode()}")

    def freq_to_closest_c_scale_note(self, freq):
        """Find the closest note in C major scale to the given frequency"""
        if freq <= 0:
            return self.c_scale_freqs[0] * 2  # Default to C4

        # Find the closest frequency in our C scale
        closest_freq = min(self.c_scale_freqs, key=lambda x: abs(x - freq))
        return closest_freq * 2

    async def autotune_audio(self, input_path: str, output_path: str):
        """Apply autotune: pitch up one octave + quantize to C major scale"""
        logger.info(f"Applying autotune (pitch up + C scale): {input_path} -> {output_path}")

        try:
            import librosa
            import soundfile as sf

            # Load audio
            audio, sr = librosa.load(input_path, sr=None)
            original_size = len(audio)
            logger.info(f"Loaded audio: {original_size} samples at {sr}Hz")

            # autotuned = librosa.effects.pitch_shift(audio, sr=sr, n_steps=0)  # 12 semitones = 1 octave
            logger.info("Pitch shifted up by one octave")

            # Save the autotuned audio
            sf.write(output_path, audio, sr)

            output_size = Path(output_path).stat().st_size
            logger.info(f"Autotune complete: {output_path} ({output_size} bytes)")

        except ImportError as e:
            logger.error(f"Missing audio processing libraries: {e}")
            # Fallback: just copy the file if librosa not available
            import shutil

            shutil.copy2(input_path, output_path)
            logger.warning("Autotune skipped - copied original file")
        except Exception as e:
            logger.error(f"Autotune processing failed: {e}")
            # Fallback: copy original file
            import shutil

            shutil.copy2(input_path, output_path)
            logger.warning("Autotune failed - copied original file")

    async def convert_with_rvc(self, input_path: str, output_path: str):
        """Convert using RVC - CPU mode only"""
        logger.info(f"Converting audio with rvc-python RVC (CPU mode): {input_path}")

        input_size = Path(input_path).stat().st_size
        logger.info(f"Input file size: {input_size} bytes")

        # Simple RVC API call
        self.rvc_instance.infer_file(input_path, output_path)

        output_size = Path(output_path).stat().st_size
        logger.info(f"rvc-python RVC conversion complete: {output_path} ({output_size} bytes)")

    async def apply_pyworld_vocoder(self, input_path: str, output_path: str):
        """Apply pyworld vocoder to modulate voice to C scale and remove airiness"""
        logger.info(f"Applying pyworld vocoder: {input_path} -> {output_path}")

        try:
            from pyworld import harvest, cheaptrick, d4c, synthesize
            import librosa
            import soundfile as sf
            import numpy as np
            from scipy import signal

            # Load audio
            audio, sr = librosa.load(input_path, sr=None)
            logger.info(f"Loaded audio for pyworld processing: {len(audio)} samples at {sr}Hz")

            # Convert to float64 for pyworld
            audio_float64 = audio.astype(np.float64)

            # Extract acoustic features
            f0, time_axis = harvest(audio_float64, sr, frame_period=5.0)
            sp = cheaptrick(audio_float64, f0, time_axis, sr)
            ap = d4c(audio_float64, f0, time_axis, sr)

            # Find min and max frequencies from voiced frames
            voiced_freqs = f0[f0 > 0]
            if len(voiced_freqs) > 0:
                min_freq = np.min(voiced_freqs)
                max_freq = np.max(voiced_freqs)
                logger.info(f"Frequency range: {min_freq:.2f}Hz - {max_freq:.2f}Hz")

                # Map to C scale range (C4 to C6)
                c_min = self.c_scale_freqs[0]  # C4 = 261.63Hz
                c_max = self.c_scale_freqs[-1]  # C6 = 1046.50Hz

                # Quantize f0 to C scale by mapping the frequency range
                f0_quantized = np.zeros_like(f0)
                for i, freq in enumerate(f0):
                    if freq > 0:  # Only process voiced frames
                        # Normalize frequency to 0-1 range within original min-max
                        if max_freq > min_freq:
                            normalized = (freq - min_freq) / (max_freq - min_freq)
                        else:
                            normalized = 0.5  # Default to middle if all frequencies are the same

                        # Map to C scale index
                        c_scale_size = len(self.c_scale_freqs)
                        target_index = int(normalized * (c_scale_size - 1))
                        target_index = np.clip(target_index, 0, c_scale_size - 1)

                        f0_quantized[i] = self.c_scale_freqs[target_index]
                    else:
                        f0_quantized[i] = 0  # Unvoiced frames set to 0
            else:
                # No voiced frames, just copy original f0
                f0_quantized = f0.copy()

            # Reduce aperiodicity to remove airiness (make it more periodic)
            ap_reduced = ap * 0.6  # Stronger reduction for stricter effect

            # Smooth spectral envelope slightly for cleaner sound
            sp_smoothed = np.zeros_like(sp)
            for i in range(sp.shape[1]):
                sp_smoothed[:, i] = np.convolve(sp[:, i], np.ones(3) / 3, mode="same")

            # Synthesize audio with modified features
            y = synthesize(f0_quantized, sp_smoothed, ap_reduced, sr)

            # Apply high-pass filter to remove low-frequency rumble
            nyquist = sr / 2
            low_cutoff = 300  # 600Hz high-pass cutoff
            high_cutoff = 32000  # 8kHz low-pass cutoff

            # Ensure cutoff frequencies are valid (0 < Wn < 1)
            low_norm = min(max(low_cutoff / nyquist, 0.0001), 0.9999)
            high_norm = min(max(high_cutoff / nyquist, 0.0001), 0.9999)

            # High-pass filter
            b_high, a_high = signal.butter(4, low_norm, btype="high")
            y_filtered = signal.filtfilt(b_high, a_high, y)

            # # Low-pass filter
            # b_low, a_low = signal.butter(4, high_norm, btype="low")
            # y_filtered = signal.filtfilt(b_low, a_low, y_high)

            # Convert back to float32 and normalize
            y_filtered = y_filtered.astype(np.float32)
            y_filtered = y_filtered / np.max(np.abs(y_filtered)) * 0.9

            # Save the processed audio
            sf.write(output_path, y_filtered, sr)

            output_size = Path(output_path).stat().st_size
            logger.info(f"Pyworld vocoder processing complete: {output_path} ({output_size} bytes)")

        except ImportError as e:
            logger.error(f"Pyworld or scipy not available: {e}")
            # Fallback: copy original file
            import shutil

            shutil.copy2(input_path, output_path)
            logger.warning("Pyworld vocoder skipped - copied original file")
        except Exception as e:
            logger.error(f"Pyworld vocoder processing failed: {e}")
            # Fallback: copy original file
            import shutil

            shutil.copy2(input_path, output_path)
            logger.warning("Pyworld vocoder failed - copied original file")

    async def convert_audio_format(self, input_path: str, output_path: str, format: str):
        """Convert audio format using ffmpeg"""
        input_size = Path(input_path).stat().st_size
        logger.info(f"Converting audio format: {format} (input: {input_size} bytes)")

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-acodec",
            self.get_ffmpeg_codec(format),
            "-ar",
            "44100",
            "-ac",
            "1",
            "-b:a",
            "128k",
            output_path,
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        await process.communicate()

        output_size = Path(output_path).stat().st_size
        logger.info(f"Audio format conversion complete: {output_path} ({output_size} bytes)")

    def get_ffmpeg_codec(self, format: str) -> str:
        codec_map = {
            "mp3": "libmp3lame",
            "opus": "libopus",
            "aac": "aac",
            "flac": "flac",
            "wav": "pcm_s16le",
            "pcm": "pcm_s16le",
        }
        return codec_map.get(format, "libmp3lame")

    def get_content_type(self, format: str) -> str:
        type_map = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm",
        }
        return type_map.get(format, "audio/mpeg")

    async def synthesize(self, request: SpeechRequest) -> tuple[bytes, str]:
        """Main synthesis pipeline with autotune layer"""
        session_id = os.urandom(8).hex()
        piper_output = Path(self.temp_dir) / f"piper_{session_id}.wav"
        autotuned_output = Path(self.temp_dir) / f"autotuned_{session_id}.wav"
        rvc_output = Path(self.temp_dir) / f"rvc_{session_id}.wav"
        pyworld_output = Path(self.temp_dir) / f"pyworld_{session_id}.wav"
        final_output = Path(self.temp_dir) / f"final_{session_id}.{request.response_format}"

        logger.info(f"Starting synthesis pipeline with autotune (session: {session_id})")

        try:
            # Step 1: Command-line Piper TTS
            await self.generate_with_piper(request.input, str(piper_output))

            # Step 2: NEW - Autotune processing (pitch up + C scale)
            await self.autotune_audio(str(piper_output), str(autotuned_output))

            # Step 3: RVC conversion (CPU mode) - now using autotuned audio
            await self.convert_with_rvc(str(autotuned_output), str(rvc_output))

            await self.apply_pyworld_vocoder(str(rvc_output), str(pyworld_output))

            # Step 4: Format conversion
            await self.convert_audio_format(str(pyworld_output), str(final_output), request.response_format)

            # Read final audio
            with open(final_output, "rb") as f:
                audio_data = f.read()

            final_size = len(audio_data)
            logger.info(f"Synthesis complete with autotune: {final_size} bytes of audio data")

            content_type = self.get_content_type(request.response_format)
            return audio_data, content_type
        finally:
            # Cleanup
            for file_path in [piper_output, autotuned_output, rvc_output, final_output]:
                try:
                    if file_path.exists():
                        file_path.unlink()
                except:
                    pass


# Initialize TTS server
tts_server = TTSServer()
logger.info("TTS Server with autotune initialized successfully")


@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    """Create speech audio with autotune"""
    if not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    logger.info(f"Processing speech request with autotune: {request.input[:100]}...")

    audio_data, content_type = await tts_server.synthesize(request)

    return Response(
        content=audio_data,
        media_type=content_type,
        headers={
            "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
            "Content-Length": str(len(audio_data)),
        },
    )


@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "version": "3.1.0 (With Autotune)",
        "mode": "CPU-only",
        "features": ["autotune", "pitch_shift_octave", "c_major_scale"],
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "OpenAI-Compatible TTS Server with Autotune",
        "version": "3.1.0",
        "pipeline": [
            "1. Piper TTS generation",
            "2. Autotune (pitch up + C scale)",
            "3. RVC voice conversion",
            "4. Audio format conversion",
        ],
        "autotune_features": {
            "pitch_shift": "+1 octave",
            "scale": "C major (C, D, E, F, G, A, B)",
            "frequency_quantization": "enabled",
        },
        "endpoints": {"speech": "/v1/audio/speech", "health": "/health"},
    }


if __name__ == "__main__":
    logger.info("Starting TTS server with autotune layer...")
    logger.info("Pipeline: Piper TTS -> Autotune (pitch up + C scale) -> RVC -> Output")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
