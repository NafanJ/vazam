/**
 * useRecorder — React hook wrapping react-native-audio-recorder-player
 *
 * Manages the full record → stop → submit lifecycle, keeping callers
 * free of AudioRecorderPlayer bookkeeping.
 *
 * Usage:
 *   const { state, duration, start, stop, reset } = useRecorder();
 */

import { useCallback, useEffect, useRef, useState } from "react";
import AudioRecorderPlayer, {
  AudioEncoderAndroidType,
  AudioSourceAndroidType,
  AVEncoderAudioQualityIOSType,
  AVEncodingOption,
} from "react-native-audio-recorder-player";
import RNFS from "react-native-fs";
import type { RecordingState } from "../types";

const MAX_DURATION_MS = 15_000;   // stop automatically after 15 s

const recorder = new AudioRecorderPlayer();

export interface UseRecorderResult {
  state: RecordingState;
  duration: number;           // ms elapsed since recording started
  filePath: string | null;    // local path of the last recording
  start: () => Promise<void>;
  stop: () => Promise<string | null>; // returns file path
  reset: () => void;
  error: string | null;
}

export function useRecorder(): UseRecorderResult {
  const [state,    setState]    = useState<RecordingState>("idle");
  const [duration, setDuration] = useState(0);
  const [filePath, setFilePath] = useState<string | null>(null);
  const [error,    setError]    = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      // Cleanup on unmount
      recorder.stopRecorder().catch(() => {});
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  const start = useCallback(async () => {
    try {
      setError(null);
      setDuration(0);
      setState("recording");

      const path = `${RNFS.CachesDirectoryPath}/vazam_recording.wav`;

      await recorder.startRecorder(path, {
        AudioEncoderAndroid: AudioEncoderAndroidType.AAC,
        AudioSourceAndroid:  AudioSourceAndroidType.MIC,
        AVEncoderAudioQualityKeyIOS: AVEncoderAudioQualityIOSType.high,
        AVNumberOfChannelsKeyIOS: 1,
        AVFormatIDKeyIOS: AVEncodingOption.wav,
      });

      recorder.addRecordBackListener((e) => {
        setDuration(e.currentPosition);
      });

      // Auto-stop at max duration
      timerRef.current = setTimeout(async () => {
        await stop();
      }, MAX_DURATION_MS);

    } catch (err: any) {
      setError(err?.message ?? "Failed to start recording");
      setState("error");
    }
  }, []);

  const stop = useCallback(async (): Promise<string | null> => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }

    try {
      const result = await recorder.stopRecorder();
      recorder.removeRecordBackListener();
      setFilePath(result);
      setState("done");
      return result;
    } catch (err: any) {
      setError(err?.message ?? "Failed to stop recording");
      setState("error");
      return null;
    }
  }, []);

  const reset = useCallback(() => {
    setState("idle");
    setDuration(0);
    setFilePath(null);
    setError(null);
  }, []);

  return { state, duration, filePath, start, stop, reset, error };
}
