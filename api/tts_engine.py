"""Motor de inferencia TTS con arquitectura modular basada en principios SOLID:
- BaseSynthesizer: Abstracción de modelos neuronales base de voz (Piper ONNX / VITS).
- ToneCloner: Modulación y clonación tímbrica de hablantes en tiempo real (OpenVoice).
- PersonaManager: Gestión de arquetipos demográficos de pacientes y modulaciones emocionales.
- TTSEngine: Orquestador del pipeline de síntesis de audio de alta fidelidad."""

from dataclasses import dataclass
import io
import logging
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import torchaudio
import torchaudio.functional as AF
from piper import PiperVoice, SynthesisConfig

from config import (
    CONVERTER_CKPT_PATH,
    CONVERTER_CONFIG_PATH,
    DEFAULT_NOISE_SCALE,
    DEFAULT_NOISE_W,
    MALE_AR_SE_PATH,
    MALE_BASE_SE_PATH,
    MAX_SPEED,
    MIN_SPEED,
    PIPER_CONFIG_PATH,
    PIPER_MALE_CONFIG_PATH,
    PIPER_MALE_MODEL_PATH,
    PIPER_MODEL_PATH,
    SAMPLE_RATE,
    VOICES_CATALOG_CSV,
    VOICES_DIR,
)

logger = logging.getLogger("tts_engine")


@dataclass(frozen=True)
class PersonaConfig:
    """Configuración acústica de un arquetipo o paciente."""
    voice_id: int
    name: str
    gender: str
    model_type: str  # "female" o "male"
    speed_base: float = 1.0
    noise_scale: float = DEFAULT_NOISE_SCALE
    noise_w: float = DEFAULT_NOISE_W
    use_cloner: bool = False


@dataclass(frozen=True)
class EmotionConfig:
    """Modificador acústico por estado emocional o síntoma."""
    speed_factor: float = 1.0
    noise_scale_mult: float = 1.0
    noise_w_mult: float = 1.0
    pitch_offset: float = 0.0


class PersonaManager:
    """Gestiona los arquetipos de pacientes y la resolución de parámetros prosódicos."""

    DEFAULT_PERSONAS: Dict[int, PersonaConfig] = {
        0: PersonaConfig(0, "daniela", "female", "female", speed_base=1.0, noise_scale=0.667, noise_w=0.8, use_cloner=False),
        1: PersonaConfig(1, "martin", "male", "male", speed_base=1.0, noise_scale=0.667, noise_w=0.8, use_cloner=True),
        2: PersonaConfig(2, "marta", "female_elderly", "female", speed_base=0.88, noise_scale=0.68, noise_w=0.82, use_cloner=False),
        3: PersonaConfig(3, "roberto", "male_elderly", "male", speed_base=0.86, noise_scale=0.68, noise_w=0.82, use_cloner=True),
        4: PersonaConfig(4, "sofia", "female_young", "female", speed_base=1.06, noise_scale=0.667, noise_w=0.78, use_cloner=False),
        5: PersonaConfig(5, "lucas", "male_young", "male", speed_base=1.06, noise_scale=0.667, noise_w=0.78, use_cloner=True),
    }

    EMOTIONS: Dict[str, EmotionConfig] = {
        "neutral": EmotionConfig(speed_factor=1.0, noise_scale_mult=1.0, noise_w_mult=1.0, pitch_offset=0.0),
        "pain": EmotionConfig(speed_factor=0.85, noise_scale_mult=1.15, noise_w_mult=1.20, pitch_offset=0.0),
        "worried": EmotionConfig(speed_factor=1.10, noise_scale_mult=1.10, noise_w_mult=0.90, pitch_offset=0.0),
        "annoyed": EmotionConfig(speed_factor=1.05, noise_scale_mult=1.05, noise_w_mult=0.80, pitch_offset=0.0),
        "custom": EmotionConfig(speed_factor=1.0, noise_scale_mult=1.0, noise_w_mult=1.0, pitch_offset=0.0),
    }

    def __init__(self, catalog_csv: Path = VOICES_CATALOG_CSV):
        self.catalog_csv = catalog_csv
        self._personas: Dict[int, PersonaConfig] = dict(self.DEFAULT_PERSONAS)
        self.catalog = self._load_catalog_dataframe()

    def _load_catalog_dataframe(self) -> pd.DataFrame:
        """Carga el catálogo de voces con fallback seguro."""
        if self.catalog_csv.exists():
            try:
                return pd.read_csv(self.catalog_csv).set_index("voice_id")
            except Exception as e:
                logger.warning(f"Error leyendo catálogo CSV: {e}. Usando catálogo por defecto.")
        
        # Fallback desde DEFAULT_PERSONAS
        rows = [
            {"voice_id": p.voice_id, "gender": p.gender, "name": p.name, "description": f"Paciente {p.name.capitalize()}"}
            for p in self._personas.values()
        ]
        return pd.DataFrame(rows).set_index("voice_id")

    def get_persona(self, voice_id: int) -> PersonaConfig:
        """Obtiene la configuración de la persona (fallback a la voz 0 si no existe)."""
        return self._personas.get(voice_id, self._personas[0])

    def get_emotion(self, emotion: str) -> EmotionConfig:
        """Obtiene la configuración de emoción (fallback a neutral)."""
        return self.EMOTIONS.get(emotion.lower(), self.EMOTIONS["neutral"])

    def is_valid_voice(self, voice_id: int) -> bool:
        return voice_id in self._personas or voice_id in self.catalog.index

    def list_voices(self) -> pd.DataFrame:
        return self.catalog.reset_index()


class ToneCloner:
    """Maneja la clonación tímbrica de hablantes mediante OpenVoice."""

    def __init__(self):
        self.converter = None
        self.src_se = None
        self.tgt_se = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._initialize()

    def _initialize(self):
        """Carga perezosa del convertidor y los vectores de timbre."""
        try:
            if (
                CONVERTER_CONFIG_PATH.exists()
                and CONVERTER_CKPT_PATH.exists()
                and MALE_AR_SE_PATH.exists()
                and MALE_BASE_SE_PATH.exists()
            ):
                from openvoice.api import ToneColorConverter

                conv = ToneColorConverter(str(CONVERTER_CONFIG_PATH), device=self.device)
                conv.load_ckpt(str(CONVERTER_CKPT_PATH))
                self.converter = conv
                self.src_se = torch.load(MALE_BASE_SE_PATH, map_location=self.device)
                self.tgt_se = torch.load(MALE_AR_SE_PATH, map_location=self.device)
                logger.info(f"ToneCloner inicializado exitosamente en {self.device}.")
        except Exception as e:
            logger.warning(f"ToneCloner no disponible ({e}). Se usará síntesis neuronal directa.")

    def clone_timbre(self, audio_buffer: io.BytesIO, tau: float = 0.3) -> io.BytesIO:
        """Convierte el timbre del audio al del locutor argentino objetivo."""
        if self.converter is None or self.src_se is None or self.tgt_se is None:
            return audio_buffer

        try:
            audio_buffer.seek(0)
            wav_converted, sr = self.converter.convert(
                audio_buffer,
                src_se=self.src_se,
                tgt_se=self.tgt_se,
                tau=tau,
            )
            out_buf = io.BytesIO()
            import soundfile as sf
            sf.write(out_buf, wav_converted.squeeze().cpu().numpy(), sr, format="WAV")
            out_buf.seek(0)
            return out_buf
        except Exception as e:
            logger.error(f"Error en clonación de timbre: {e}")
            audio_buffer.seek(0)
            return audio_buffer


class BaseSynthesizer:
    """Carga y gestiona los modelos neuronales base de Piper TTS."""

    def __init__(self):
        self.female_voice = self._load_model(PIPER_MODEL_PATH, PIPER_CONFIG_PATH)
        self.male_voice = self._load_model(PIPER_MALE_MODEL_PATH, PIPER_MALE_CONFIG_PATH, fallback=self.female_voice)

    def _load_model(self, model_path: Path, config_path: Path, fallback: Optional[PiperVoice] = None) -> PiperVoice:
        """Carga un modelo PiperVoice asegurando su presencia."""
        if not model_path.exists() or not config_path.exists():
            try:
                import sys
                sys.path.insert(0, str(VOICES_DIR))
                from download_voices import ensure_voices
                ensure_voices()
            except Exception as e:
                logger.warning(f"No se pudieron descargar voces automáticamente: {e}")

        if model_path.exists() and config_path.exists():
            return PiperVoice.load(str(model_path), config_path=str(config_path))
        
        if fallback is not None:
            return fallback
        
        raise RuntimeError(f"Modelo requerido no encontrado: {model_path}")

    def get_model(self, model_type: str) -> PiperVoice:
        """Retorna el modelo neuronal según el género."""
        return self.male_voice if model_type == "male" else self.female_voice


class TTSEngine:
    """Orquestador principal de síntesis TTS de alto rendimiento y arquitectura extensible."""

    def __init__(self):
        self.persona_mgr = PersonaManager()
        self.synthesizer = BaseSynthesizer()
        self.cloner = ToneCloner()

    @property
    def catalog(self) -> pd.DataFrame:
        """Compatibilidad con endpoints existentes y tests."""
        return self.persona_mgr.catalog

    @catalog.setter
    def catalog(self, val: pd.DataFrame):
        self.persona_mgr.catalog = val

    def is_valid_voice(self, voice_id: int) -> bool:
        return self.persona_mgr.is_valid_voice(voice_id)

    def list_voices(self) -> pd.DataFrame:
        return self.persona_mgr.list_voices()

    def synthesize(
        self,
        text: str,
        voice_id: int = 0,
        speed: float = 1.0,
        style_strength: float = 1.0,
        emotion: str = "neutral",
        pitch_shift: Optional[float] = None,
    ) -> bytes:
        """Pipeline de síntesis: Selección de modelo -> Generación Base -> Clonación Tímbrica -> Modulación DSP."""
        persona = self.persona_mgr.get_persona(voice_id)
        emotion_mod = self.persona_mgr.get_emotion(emotion)

        # 1. Selección de modelo neuronal base
        model = self.synthesizer.get_model(persona.model_type)

        # 2. Parámetros de articulación prosódica
        effective_speed = speed * persona.speed_base * emotion_mod.speed_factor
        length_scale = 1.0 / max(MIN_SPEED, min(MAX_SPEED, effective_speed))

        noise_scale = persona.noise_scale * emotion_mod.noise_scale_mult * max(0.4, min(1.6, style_strength))
        noise_w_scale = persona.noise_w * emotion_mod.noise_w_mult * max(0.4, min(1.6, style_strength))

        syn_config = SynthesisConfig(
            speaker_id=None,
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w_scale,
            normalize_audio=True,
            volume=1.0,
        )

        # 3. Síntesis base de audio
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            model.synthesize_wav(
                text=text,
                wav_file=wav_file,
                syn_config=syn_config,
            )

        buffer.seek(0)

        # 4. Clonación tímbrica (para arquetipos masculinos del dataset argentino)
        if persona.use_cloner:
            buffer = self.cloner.clone_timbre(buffer)

        # 5. Modulación tonal DSP si fue solicitada explícitamente
        total_pitch_shift = emotion_mod.pitch_offset + (pitch_shift if pitch_shift is not None else 0.0)
        if abs(total_pitch_shift) > 0.05:
            wav_tensor, sr = torchaudio.load(buffer)
            wav_tensor = AF.pitch_shift(
                wav_tensor,
                sample_rate=sr,
                n_steps=total_pitch_shift,
                n_fft=2048,
                win_length=1024,
                hop_length=256,
            )
            out_buf = io.BytesIO()
            torchaudio.save(out_buf, wav_tensor, sr, format="wav")
            out_buf.seek(0)
            return out_buf.read()

        return buffer.read()


engine = TTSEngine()
