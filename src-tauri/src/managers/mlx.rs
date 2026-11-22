//! An existing MLX Whisper snapshot, owned by the user or Hugging Face.
use super::model::{ModelInfo, ModelSource};
use crate::engine_supervisor::protocol::{LoadedInfo, Request, Response};
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::process::Command;
use std::sync::mpsc::Sender;
use tauri::AppHandle;
use transcribe_cpp::{Capabilities, TimestampKind, Transcript};

pub const MLX_MODEL_ID: &str = "shared:whisper-mlx";

#[derive(Debug, Clone, Deserialize)]
pub struct MlxConfig {
    python: PathBuf,
    pub snapshot: PathBuf,
    /// An existing worker can supply its own language policy.
    script: Option<PathBuf>,
}

impl MlxConfig {
    pub fn read(app: &AppHandle) -> Result<Self> {
        ensure!(
            cfg!(all(target_os = "macos", target_arch = "aarch64")),
            "MLX needs Apple Silicon"
        );
        let path = crate::portable::app_data_dir(app)?.join("whisper-mlx.json");
        let config: Self = serde_json::from_slice(&std::fs::read(&path)?)
            .with_context(|| format!("Reading {}", path.display()))?;
        ensure!(
            config.python.is_absolute() && config.python.is_file(),
            "Missing MLX Python interpreter"
        );
        ensure!(
            config.snapshot.is_absolute()
                && config.snapshot.join("config.json").is_file()
                && config.snapshot.join("weights.safetensors").is_file(),
            "Missing MLX Whisper snapshot"
        );
        if let Some(script) = &config.script {
            ensure!(
                script.is_absolute() && script.is_file(),
                "Missing MLX worker script"
            );
        }
        Ok(config)
    }

    fn languages(&self) -> Vec<String> {
        if self.script.is_some() {
            // Existing custom-worker configurations were for bilingual dictation.
            return vec!["tr".into(), "en".into()];
        }
        crate::catalog::CATALOG
            .iter()
            .find(|m| {
                m.id.starts_with("handy-computer/whisper-large-v3-turbo-gguf/")
            })
            .and_then(|m| m.caps.languages.clone())
            .unwrap_or_default()
    }

    pub fn model_info(&self) -> ModelInfo {
        ModelInfo {
            id: MLX_MODEL_ID.into(),
            name: "Whisper Large v3 Turbo (MLX)".into(),
            description: "Uses an existing local MLX Whisper snapshot.".into(),
            filename: "weights.safetensors".into(),
            source: ModelSource::Local,
            size_mb: self
                .snapshot
                .join("weights.safetensors")
                .metadata()
                .map_or(0, |m| m.len() / (1024 * 1024)),
            is_downloaded: true,
            is_downloading: false,
            partial_size: 0,
            accuracy_score: 0.0,
            speed_score: 0.0,
            supports_translation: false,
            is_recommended: false,
            supported_languages: self.languages(),
            supports_language_selection: true,
            is_custom: true,
            supports_streaming: false,
            supports_language_detection: true,
        }
    }

    pub(crate) fn command(&self) -> Command {
        let mut cmd = Command::new(&self.python);
        cmd.arg("-I");
        if let Some(script) = &self.script {
            cmd.arg(script);
        } else {
            cmd.arg("-c").arg(include_str!("mlx_worker.py"));
        }
        cmd.arg(&self.snapshot).env("HF_HUB_OFFLINE", "1");
        cmd
    }

    pub(crate) fn loaded_info(&self) -> LoadedInfo {
        LoadedInfo {
            arch: "whisper".into(),
            variant: "large-v3-turbo".into(),
            backend: "MLX".into(),
            device: "Apple GPU".into(),
            on_gpu: true,
            capabilities: Capabilities {
                native_sample_rate: 16_000,
                languages: self.languages(),
                translate_target_languages: vec![],
                max_timestamp_kind: TimestampKind::None,
                supports_language_detect: true,
                supports_translate: false,
                supports_streaming: false,
                supports_spec_decode: false,
                max_audio_ms: 0,
            },
            supports_initial_prompt: true,
            supports_long_form: true,
            supports_vocabulary: true,
        }
    }
}

pub(crate) fn encode_run(request: &Request, pcm: &[f32]) -> Result<Vec<u8>> {
    let Request::Run { options } = request else {
        anyhow::bail!("MLX Whisper only supports batch transcription");
    };
    let mut frame = serde_json::to_vec(&serde_json::json!({
        "samples": pcm.len(),
        "language": options.language.as_deref().unwrap_or("auto"),
        "custom_words": options.vocabulary,
    }))?;
    frame.push(b'\n');
    frame.reserve(std::mem::size_of_val(pcm));
    for sample in pcm {
        frame.extend_from_slice(&sample.to_le_bytes());
    }
    Ok(frame)
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Reply {
    Text {
        text: String,
        language: Option<String>,
    },
    Error {
        error: String,
    },
}

pub(crate) fn read_replies(output: impl Read, tx: Sender<Response>) {
    for (index, line) in BufReader::new(output).lines().enumerate() {
        let response = match line
            .and_then(|line| serde_json::from_str(&line).map_err(std::io::Error::other))
        {
            Ok(Reply::Text { .. }) if index == 0 => Response::Ok,
            Ok(Reply::Text { text, language }) => Response::Transcript(Transcript {
                raw_text: text.clone(),
                text,
                language,
                ..Default::default()
            }),
            Ok(Reply::Error { error }) => Response::Error(error),
            Err(error) => {
                let _ = tx.send(Response::Error(format!(
                    "Reading MLX worker reply: {error}"
                )));
                break;
            }
        };
        if tx.send(response).is_err() {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use transcribe_cpp::RunOptions;

    #[test]
    fn sends_auto_language_and_little_endian_pcm() {
        let request = Request::Run {
            options: RunOptions {
                vocabulary: vec!["İstanbul".into()],
                ..Default::default()
            },
        };
        let frame = encode_run(&request, &[0.25, -1.0]).unwrap();
        let newline = frame.iter().position(|byte| *byte == b'\n').unwrap();
        let header: serde_json::Value = serde_json::from_slice(&frame[..newline]).unwrap();
        assert_eq!(
            header,
            serde_json::json!({"samples": 2, "language": "auto", "custom_words": ["İstanbul"]})
        );
        assert_eq!(
            frame[newline + 1..],
            [0.25f32.to_le_bytes(), (-1.0f32).to_le_bytes()].concat()
        );
    }

    #[test]
    fn distinguishes_ready_empty_transcript_and_errors() {
        let replies = b"{\"text\":\"\"}\n{\"text\":\"\"}\n{\"text\":\"hello\",\"language\":\"en\"}\n{\"error\":\"failed\"}\n";
        let (tx, rx) = mpsc::channel();
        read_replies(&replies[..], tx);
        assert!(matches!(rx.recv().unwrap(), Response::Ok));
        assert!(matches!(rx.recv().unwrap(), Response::Transcript(t) if t.text.is_empty()));
        assert!(
            matches!(rx.recv().unwrap(), Response::Transcript(t) if t.language.as_deref() == Some("en"))
        );
        assert!(matches!(rx.recv().unwrap(), Response::Error(e) if e == "failed"));
    }
}
