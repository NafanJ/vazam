/**
 * ResultCard — displays one identification match
 *
 * Shows actor name, character name, a confidence bar, and a match-level badge.
 * Tapping navigates to the actor's full profile.
 */

import React from "react";
import {
  Image,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import type { IdentificationMatch } from "../types";

interface ResultCardProps {
  match: IdentificationMatch;
  rank: number;
  onPress?: (actorId: number) => void;
}

const BADGE_COLORS: Record<string, string> = {
  confident: "#34C759",
  possible:  "#FF9500",
  none:      "#8E8E93",
};

const BADGE_LABELS: Record<string, string> = {
  confident: "Match",
  possible:  "Possible",
  none:      "Unlikely",
};

export const ResultCard: React.FC<ResultCardProps> = ({ match, rank, onPress }) => {
  const badgeColor = BADGE_COLORS[match.match_level] ?? BADGE_COLORS.none;
  const badgeLabel = BADGE_LABELS[match.match_level] ?? "Unknown";
  const confidencePct = Math.round(match.confidence * 100);

  return (
    <TouchableOpacity
      style={styles.card}
      activeOpacity={0.7}
      onPress={() => onPress?.(match.actor_id)}
      accessibilityLabel={`${match.actor_name} as ${match.character_name}, ${confidencePct}% confidence`}
    >
      {/* Rank badge */}
      <View style={styles.rankBadge}>
        <Text style={styles.rankText}>#{rank}</Text>
      </View>

      {/* Main content */}
      <View style={styles.content}>
        <View style={styles.header}>
          <Text style={styles.actorName} numberOfLines={1}>
            {match.actor_name}
          </Text>
          <View style={[styles.matchBadge, { backgroundColor: badgeColor }]}>
            <Text style={styles.matchBadgeText}>{badgeLabel}</Text>
          </View>
        </View>

        <Text style={styles.characterName} numberOfLines={1}>
          as {match.character_name}
        </Text>

        {/* Confidence bar */}
        <View style={styles.barTrack}>
          <View
            style={[
              styles.barFill,
              {
                width:           `${confidencePct}%`,
                backgroundColor: badgeColor,
              },
            ]}
          />
        </View>
        <Text style={styles.confidenceLabel}>{confidencePct}% confidence</Text>
      </View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  card: {
    flexDirection:  "row",
    alignItems:     "center",
    backgroundColor: "#1C1C1E",
    borderRadius:   12,
    marginHorizontal: 16,
    marginVertical:  6,
    padding:        14,
    shadowColor:    "#000",
    shadowOpacity:  0.25,
    shadowRadius:   8,
    shadowOffset:   { width: 0, height: 2 },
    elevation:      3,
  },
  rankBadge: {
    width:          32,
    height:         32,
    borderRadius:   16,
    backgroundColor: "#2C2C2E",
    alignItems:     "center",
    justifyContent: "center",
    marginRight:    12,
  },
  rankText: {
    color:      "#8E8E93",
    fontSize:   13,
    fontWeight: "600",
  },
  content: {
    flex: 1,
  },
  header: {
    flexDirection:  "row",
    alignItems:     "center",
    justifyContent: "space-between",
    marginBottom:   2,
  },
  actorName: {
    color:      "#FFFFFF",
    fontSize:   17,
    fontWeight: "700",
    flex:       1,
    marginRight: 8,
  },
  matchBadge: {
    borderRadius: 6,
    paddingHorizontal: 8,
    paddingVertical:   2,
  },
  matchBadgeText: {
    color:      "#FFFFFF",
    fontSize:   12,
    fontWeight: "600",
  },
  characterName: {
    color:       "#AEAEB2",
    fontSize:    14,
    marginBottom: 8,
  },
  barTrack: {
    height:          4,
    backgroundColor: "#3A3A3C",
    borderRadius:    2,
    marginBottom:    4,
  },
  barFill: {
    height:       4,
    borderRadius: 2,
  },
  confidenceLabel: {
    color:    "#8E8E93",
    fontSize: 12,
  },
});
