/**
 * MangoVoice — Voice recorder hook.
 * Uses MediaRecorder + Web Audio API for recording + amplitude.
 * Simple VAD: stops recording after N seconds of silence.
 */
"use client";

import { useCallback, useEffect, useRef, useState } from "react";

interface UseVoiceRecorderOptions {
  maxSeconds?: number;
  silenceTimeout?: number; // ms of silence before auto-stop
  silenceThreshold?: number; // amplitude below which = silence
  onAmplitude?: (amp: number) => void;
}

interface UseVoiceRecorderReturn {
  isRecording: boolean;
  amplitude: number;
  start: () => Promise<void>;
  stop: () => Promise<Blob | null>;
  error: string | null;
  isSupported: boolean;
}

function getBestSupportedMimeType(): string | undefined {
  if (typeof MediaRecorder === "undefined") return undefined;
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/aac",
    "audio/ogg;codecs=opus",
    "audio/ogg",
    "audio/wav",
  ];
  for (const candidate of candidates) {
    try {
      if (MediaRecorder.isTypeSupported(candidate)) {
        return candidate;
      }
    } catch {
      // ignore
    }
  }
  return undefined;
}

export function useVoiceRecorder({
  maxSeconds = 25,
  silenceTimeout = 2500,
  silenceThreshold = 0.02,
  onAmplitude,
}: UseVoiceRecorderOptions = {}): UseVoiceRecorderReturn {
  const [isRecording, setIsRecording] = useState(false);
  const [amplitude, setAmplitude] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const isRecordingRef = useRef(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animFrameRef = useRef<number>(0);
  const maxTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const silenceTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const resolveRef = useRef<((blob: Blob | null) => void) | null>(null);

  const isSupported =
    typeof navigator !== "undefined" &&
    typeof navigator.mediaDevices !== "undefined" &&
    typeof MediaRecorder !== "undefined";

  const cleanup = useCallback(() => {
    isRecordingRef.current = false;
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    if (maxTimeoutRef.current) clearTimeout(maxTimeoutRef.current);
    if (silenceTimeoutRef.current) clearTimeout(silenceTimeoutRef.current);
    if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
      audioCtxRef.current.close().catch(() => {});
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
    }
    streamRef.current = null;
    audioCtxRef.current = null;
    analyserRef.current = null;
    mediaRecorderRef.current = null;
  }, []);

  useEffect(() => {
    return () => {
      cleanup();
    };
  }, [cleanup]);

  const start = useCallback(async () => {
    if (!isSupported) {
      const msg = "Microphone not supported in this browser";
      setError(msg);
      throw new Error(msg);
    }

    setError(null);
    chunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000,
        },
      });
      streamRef.current = stream;

      // Audio context for amplitude analysis
      try {
        const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        const audioCtx = new AudioCtx({ sampleRate: 16000 });
        audioCtxRef.current = audioCtx;
        const source = audioCtx.createMediaStreamSource(stream);
        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 256;
        source.connect(analyser);
        analyserRef.current = analyser;
      } catch (audioCtxErr) {
        console.warn("AudioContext initialization warning:", audioCtxErr);
      }

      // MediaRecorder with robust cross-browser fallback
      const mimeType = getBestSupportedMimeType();
      let recorder: MediaRecorder;
      try {
        recorder = mimeType
          ? new MediaRecorder(stream, { mimeType })
          : new MediaRecorder(stream);
      } catch {
        // Fallback to basic MediaRecorder without mime options
        recorder = new MediaRecorder(stream);
      }

      mediaRecorderRef.current = recorder;
      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.start(250); // collect in 250ms chunks
      isRecordingRef.current = true;
      setIsRecording(true);

      // Max duration timeout
      maxTimeoutRef.current = setTimeout(() => {
        if (recorder.state === "recording") recorder.stop();
      }, maxSeconds * 1000);

      // Amplitude polling + VAD
      if (analyserRef.current) {
        const analyser = analyserRef.current;
        const dataArr = new Uint8Array(analyser.frequencyBinCount);
        let silenceStart: number | null = null;

        const pollAmplitude = () => {
          if (!analyserRef.current || !isRecordingRef.current) return;
          analyser.getByteTimeDomainData(dataArr);
          let sum = 0;
          for (let i = 0; i < dataArr.length; i++) {
            const v = (dataArr[i] - 128) / 128;
            sum += v * v;
          }
          const rms = Math.sqrt(sum / dataArr.length);
          setAmplitude(rms);
          onAmplitude?.(rms);

          // VAD: track silence duration
          if (rms < silenceThreshold) {
            if (silenceStart === null) silenceStart = Date.now();
            if (Date.now() - silenceStart > silenceTimeout) {
              if (recorder.state === "recording") {
                recorder.stop();
                return;
              }
            }
          } else {
            silenceStart = null;
          }

          animFrameRef.current = requestAnimationFrame(pollAmplitude);
        };
        animFrameRef.current = requestAnimationFrame(pollAmplitude);
      }
    } catch (err: unknown) {
      const msg =
        err instanceof Error
          ? err.name === "NotAllowedError" || err.name === "PermissionDeniedError"
            ? "Microphone permission denied. Please allow microphone access in your browser."
            : err.message
          : "Could not access microphone";
      setError(msg);
      isRecordingRef.current = false;
      setIsRecording(false);
      cleanup();
      throw new Error(msg);
    }
  }, [isSupported, maxSeconds, silenceTimeout, silenceThreshold, onAmplitude, cleanup]);


  const stop = useCallback(async (): Promise<Blob | null> => {
    const recorder = mediaRecorderRef.current;
    if (!recorder || recorder.state === "inactive") {
      setIsRecording(false);
      cleanup();
      return null;
    }

    return new Promise((resolve) => {
      resolveRef.current = resolve;
      recorder.onstop = () => {
        const mime = recorder.mimeType || getBestSupportedMimeType() || "audio/webm";
        const blob = new Blob(chunksRef.current, { type: mime });
        cleanup();
        setIsRecording(false);
        setAmplitude(0);
        resolve(blob.size > 0 ? blob : null);
      };
      if (recorder.state === "recording") {
        recorder.stop();
      }
    });
  }, [cleanup]);

  return { isRecording, amplitude, start, stop, error, isSupported };
}

