/**
 * HomeScreen — main "Listen" screen
 *
 * User flow:
 *   1. (Optional) Select a show to narrow the search
 *   2. Tap the record button → records up to 15 s
 *   3. Tap again → stops recording → submits to API
 *   4. On success → navigate to ResultsScreen
 */

import React, { useCallback, useState } from "react";
import {
  Alert,
  Animated,
  SafeAreaView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { useNavigation } from "@react-navigation/native";
import type { NativeStackNavigationProp } from "@react-navigation/native-stack";

import { identify } from "../api/vazam";
import { RecordButton } from "../components/RecordButton";
import { useRecorder } from "../hooks/useRecorder";
import type { RootStackParamList } from "../types";

type Nav = NativeStackNavigationProp<RootStackParamList, "Home">;

const STATUS_LABELS: Record<string, string> = {
  idle:       "Tap to identify a voice",
  recording:  "Listening…",
  processing: "Identifying…",
  done:       "Done",
  error:      "Something went wrong",
};

export default function HomeScreen(): React.JSX.Element {
  const navigation = useNavigation<Nav>();
  const { state, duration, start, stop, reset, error } = useRecorder();
  const [selectedShowId, setSelectedShowId] = useState<number | undefined>();
  const [selectedShowTitle, setSelectedShowTitle] = useState<string>("");

  const durationLabel = (() => {
    const s = Math.floor(duration / 1000);
    return `${s}s / 15s`;
  })();

  const handlePress = useCallback(async () => {
    if (state === "idle" || state === "done" || state === "error") {
      if (state !== "idle") reset();
      await start();
    } else if (state === "recording") {
      const filePath = await stop();
      if (!filePath) return;

      // Brief state shows "processing" visually; API call happens here
      try {
        const response = await identify({
          audioPath:  filePath,
          isolate:    true,   // always isolate from real-world audio
          showId:     selectedShowId,
          topK:       5,
        });

        navigation.navigate("Results", { results: response.results });
      } catch (err: any) {
        Alert.alert(
          "Identification failed",
          err?.response?.data?.detail ?? err?.message ?? "Unknown error"
        );
        reset();
      }
    }
  }, [state, start, stop, reset, selectedShowId, navigation]);

  const clearShow = useCallback(() => {
    setSelectedShowId(undefined);
    setSelectedShowTitle("");
  }, []);

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.logo}>Vazam</Text>
        <Text style={styles.tagline}>Shazam for Voice Actors</Text>
      </View>

      {/* Show selector */}
      <View style={styles.showRow}>
        {selectedShowTitle ? (
          <TouchableOpacity style={styles.showChip} onPress={clearShow}>
            <Text style={styles.showChipText} numberOfLines={1}>
              {selectedShowTitle}  ✕
            </Text>
          </TouchableOpacity>
        ) : (
          <TouchableOpacity
            style={styles.showSelectBtn}
            onPress={() => navigation.navigate("ShowSearch")}
          >
            <Text style={styles.showSelectText}>+ Filter by show</Text>
          </TouchableOpacity>
        )}
      </View>

      {/* Record button area */}
      <View style={styles.recordArea}>
        <RecordButton state={state} onPress={handlePress} size={90} />

        <Text style={styles.statusLabel}>
          {STATUS_LABELS[state] ?? STATUS_LABELS.idle}
        </Text>

        {state === "recording" && (
          <Text style={styles.durationLabel}>{durationLabel}</Text>
        )}

        {error && (
          <Text style={styles.errorText}>{error}</Text>
        )}
      </View>

      {/* Hint */}
      {state === "idle" && (
        <Text style={styles.hint}>
          Hold your phone near the audio source for best results.
          {"\n"}Works on TV, streaming, or live events.
        </Text>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex:            1,
    backgroundColor: "#000000",
    alignItems:      "center",
  },
  header: {
    marginTop:  40,
    alignItems: "center",
  },
  logo: {
    color:      "#FFFFFF",
    fontSize:   36,
    fontWeight: "800",
    letterSpacing: 1,
  },
  tagline: {
    color:     "#8E8E93",
    fontSize:  14,
    marginTop: 4,
  },
  showRow: {
    marginTop:    24,
    paddingHorizontal: 24,
    alignSelf: "stretch",
    alignItems: "center",
  },
  showChip: {
    backgroundColor: "#1C1C1E",
    borderRadius:    20,
    paddingHorizontal: 16,
    paddingVertical:   8,
    maxWidth:        260,
  },
  showChipText: {
    color:    "#007AFF",
    fontSize: 14,
  },
  showSelectBtn: {
    borderWidth:   1,
    borderColor:   "#3A3A3C",
    borderRadius:  20,
    paddingHorizontal: 16,
    paddingVertical:   8,
  },
  showSelectText: {
    color:    "#8E8E93",
    fontSize: 14,
  },
  recordArea: {
    flex:           1,
    alignItems:     "center",
    justifyContent: "center",
    gap:            20,
  },
  statusLabel: {
    color:      "#FFFFFF",
    fontSize:   18,
    fontWeight: "600",
    marginTop:  16,
  },
  durationLabel: {
    color:    "#8E8E93",
    fontSize: 14,
  },
  errorText: {
    color:    "#FF3B30",
    fontSize: 13,
    textAlign: "center",
    marginHorizontal: 32,
  },
  hint: {
    color:     "#636366",
    fontSize:  13,
    textAlign: "center",
    marginHorizontal: 40,
    marginBottom: 40,
    lineHeight: 20,
  },
});
