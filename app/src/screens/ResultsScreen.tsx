/**
 * ResultsScreen — ranked list of identification results
 *
 * Receives `results: IdentificationMatch[]` from navigation params.
 * Tapping a result navigates to the actor's full profile.
 */

import React from "react";
import {
  FlatList,
  SafeAreaView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { useNavigation, useRoute } from "@react-navigation/native";
import type {
  NativeStackNavigationProp,
  NativeStackScreenProps,
} from "@react-navigation/native-stack";

import { ResultCard } from "../components/ResultCard";
import type { IdentificationMatch, RootStackParamList } from "../types";

type Props    = NativeStackScreenProps<RootStackParamList, "Results">;
type Nav      = NativeStackNavigationProp<RootStackParamList, "Results">;

export default function ResultsScreen(): React.JSX.Element {
  const navigation = useNavigation<Nav>();
  const { params } = useRoute<Props["route"]>();
  const { results } = params;

  const handleActorPress = (actorId: number, actorName: string) => {
    navigation.navigate("ActorProfile", { actorId, actorName });
  };

  const noResults = results.length === 0;
  const topResult = results[0];
  const isConfident = topResult?.match_level === "confident";

  return (
    <SafeAreaView style={styles.container}>
      {/* Summary header */}
      <View style={styles.summaryHeader}>
        {noResults ? (
          <>
            <Text style={styles.summaryTitle}>No match found</Text>
            <Text style={styles.summarySubtitle}>
              Try a longer clip or enable voice isolation
            </Text>
          </>
        ) : isConfident ? (
          <>
            <Text style={styles.summaryTitle}>{topResult.actor_name}</Text>
            <Text style={styles.summarySubtitle}>
              as {topResult.character_name}
            </Text>
            <Text style={styles.confidencePct}>
              {Math.round(topResult.confidence * 100)}% confidence
            </Text>
          </>
        ) : (
          <>
            <Text style={styles.summaryTitle}>Possible match</Text>
            <Text style={styles.summarySubtitle}>
              Low confidence — try a longer, cleaner clip
            </Text>
          </>
        )}
      </View>

      {/* Results list */}
      {!noResults && (
        <FlatList
          data={results}
          keyExtractor={(item, i) => `${item.actor_id}-${item.character_name}-${i}`}
          renderItem={({ item, index }) => (
            <ResultCard
              match={item}
              rank={index + 1}
              onPress={(id) => handleActorPress(id, item.actor_name)}
            />
          )}
          contentContainerStyle={styles.list}
          ListHeaderComponent={
            results.length > 1 ? (
              <Text style={styles.listHeader}>All candidates</Text>
            ) : null
          }
        />
      )}

      {/* Actions */}
      <View style={styles.actions}>
        <TouchableOpacity
          style={styles.actionBtn}
          onPress={() => navigation.goBack()}
        >
          <Text style={styles.actionBtnText}>Try again</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex:            1,
    backgroundColor: "#000000",
  },
  summaryHeader: {
    alignItems:   "center",
    paddingTop:   32,
    paddingBottom: 24,
    paddingHorizontal: 24,
    borderBottomWidth: 1,
    borderBottomColor: "#1C1C1E",
  },
  summaryTitle: {
    color:       "#FFFFFF",
    fontSize:    28,
    fontWeight:  "800",
    textAlign:   "center",
    marginBottom: 4,
  },
  summarySubtitle: {
    color:     "#AEAEB2",
    fontSize:  16,
    textAlign: "center",
  },
  confidencePct: {
    color:     "#34C759",
    fontSize:  14,
    marginTop: 6,
    fontWeight: "600",
  },
  list: {
    paddingTop:    12,
    paddingBottom: 24,
  },
  listHeader: {
    color:           "#8E8E93",
    fontSize:        13,
    fontWeight:      "600",
    textTransform:   "uppercase",
    letterSpacing:   0.5,
    marginHorizontal: 16,
    marginBottom:    8,
  },
  actions: {
    padding:      20,
    borderTopWidth:  1,
    borderTopColor:  "#1C1C1E",
  },
  actionBtn: {
    backgroundColor: "#1C1C1E",
    borderRadius:    12,
    paddingVertical: 14,
    alignItems:      "center",
  },
  actionBtnText: {
    color:      "#FFFFFF",
    fontSize:   16,
    fontWeight: "600",
  },
});
