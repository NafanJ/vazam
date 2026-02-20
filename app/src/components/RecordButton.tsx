/**
 * RecordButton — animated circular tap-to-record button
 *
 * States:
 *   idle       → grey ring, mic icon
 *   recording  → red pulsing ring, stop icon
 *   processing → spinner
 *   error      → red icon, shake animation
 */

import React, { useEffect, useRef } from "react";
import {
  Animated,
  Easing,
  StyleSheet,
  TouchableOpacity,
  View,
} from "react-native";
import type { RecordingState } from "../types";

interface RecordButtonProps {
  state: RecordingState;
  onPress: () => void;
  size?: number;
}

const COLORS: Record<RecordingState, string> = {
  idle:       "#E0E0E0",
  recording:  "#FF3B30",
  processing: "#007AFF",
  done:       "#34C759",
  error:      "#FF3B30",
};

const ICONS: Partial<Record<RecordingState, string>> = {
  idle:       "●",   // mic placeholder (use vector icons in production)
  recording:  "■",
  processing: "…",
  error:      "!",
};

export const RecordButton: React.FC<RecordButtonProps> = ({
  state,
  onPress,
  size = 80,
}) => {
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const spinAnim  = useRef(new Animated.Value(0)).current;

  // Pulse animation while recording
  useEffect(() => {
    if (state === "recording") {
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, {
            toValue:         1.15,
            duration:        600,
            easing:          Easing.inOut(Easing.ease),
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnim, {
            toValue:         1,
            duration:        600,
            easing:          Easing.inOut(Easing.ease),
            useNativeDriver: true,
          }),
        ])
      ).start();
    } else {
      pulseAnim.stopAnimation();
      pulseAnim.setValue(1);
    }
  }, [state, pulseAnim]);

  // Spin animation while processing
  useEffect(() => {
    if (state === "processing") {
      Animated.loop(
        Animated.timing(spinAnim, {
          toValue:         1,
          duration:        800,
          easing:          Easing.linear,
          useNativeDriver: true,
        })
      ).start();
    } else {
      spinAnim.stopAnimation();
      spinAnim.setValue(0);
    }
  }, [state, spinAnim]);

  const spin = spinAnim.interpolate({
    inputRange:  [0, 1],
    outputRange: ["0deg", "360deg"],
  });

  const color = COLORS[state] ?? COLORS.idle;
  const icon  = ICONS[state]  ?? ICONS.idle;

  const isDisabled = state === "processing";

  return (
    <TouchableOpacity
      onPress={onPress}
      disabled={isDisabled}
      activeOpacity={0.8}
      accessibilityLabel={state === "recording" ? "Stop recording" : "Start recording"}
      accessibilityRole="button"
    >
      <Animated.View
        style={[
          styles.outer,
          {
            width:        size * 1.4,
            height:       size * 1.4,
            borderRadius: (size * 1.4) / 2,
            borderColor:  color,
            transform:    [{ scale: pulseAnim }],
          },
        ]}
      >
        <Animated.View
          style={[
            styles.inner,
            {
              width:            size,
              height:           size,
              borderRadius:     size / 2,
              backgroundColor:  color,
              transform:        state === "processing" ? [{ rotate: spin }] : [],
            },
          ]}
        >
          <Animated.Text style={[styles.icon, { fontSize: size * 0.35 }]}>
            {icon}
          </Animated.Text>
        </Animated.View>
      </Animated.View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  outer: {
    alignItems:     "center",
    justifyContent: "center",
    borderWidth:    3,
  },
  inner: {
    alignItems:     "center",
    justifyContent: "center",
  },
  icon: {
    color: "#FFFFFF",
    fontWeight: "bold",
  },
});
