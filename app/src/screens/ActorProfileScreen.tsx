/**
 * ActorProfileScreen — full voice actor profile with filmography
 */

import React, { useEffect, useState } from "react";
import {
  ActivityIndicator,
  FlatList,
  Image,
  SafeAreaView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import type { NativeStackScreenProps } from "@react-navigation/native-stack";

import { getActorProfile } from "../api/vazam";
import type { ActorProfile, FilmographyEntry, RootStackParamList } from "../types";

type Props = NativeStackScreenProps<RootStackParamList, "ActorProfile">;

function FilmographyRow({ entry }: { entry: FilmographyEntry }) {
  return (
    <View style={styles.filmRow}>
      <View style={styles.filmInfo}>
        <Text style={styles.charName} numberOfLines={1}>
          {entry.character_name}
        </Text>
        <Text style={styles.showName} numberOfLines={1}>
          {entry.show_title ?? "Unknown show"}
          {entry.year ? `  (${entry.year})` : ""}
        </Text>
      </View>
      {entry.media_type && (
        <View style={styles.typeBadge}>
          <Text style={styles.typeBadgeText}>{entry.media_type}</Text>
        </View>
      )}
    </View>
  );
}

export default function ActorProfileScreen({ route }: Props): React.JSX.Element {
  const { actorId, actorName } = route.params;
  const [profile, setProfile] = useState<ActorProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState<string | null>(null);

  useEffect(() => {
    getActorProfile(actorId)
      .then(setProfile)
      .catch((err) => setError(err?.message ?? "Failed to load profile"))
      .finally(() => setLoading(false));
  }, [actorId]);

  if (loading) {
    return (
      <SafeAreaView style={styles.centered}>
        <ActivityIndicator size="large" color="#007AFF" />
      </SafeAreaView>
    );
  }

  if (error || !profile) {
    return (
      <SafeAreaView style={styles.centered}>
        <Text style={styles.errorText}>{error ?? "Profile not available"}</Text>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <FlatList
        data={profile.filmography}
        keyExtractor={(item, i) => `${item.character_name}-${i}`}
        renderItem={({ item }) => <FilmographyRow entry={item} />}
        ListHeaderComponent={
          <View style={styles.profileHeader}>
            {profile.image_url ? (
              <Image source={{ uri: profile.image_url }} style={styles.avatar} />
            ) : (
              <View style={[styles.avatar, styles.avatarPlaceholder]}>
                <Text style={styles.avatarInitial}>
                  {profile.name.charAt(0).toUpperCase()}
                </Text>
              </View>
            )}

            <Text style={styles.name}>{profile.name}</Text>

            {profile.bio ? (
              <Text style={styles.bio} numberOfLines={4}>
                {profile.bio}
              </Text>
            ) : null}

            <View style={styles.statRow}>
              <View style={styles.stat}>
                <Text style={styles.statValue}>{profile.filmography.length}</Text>
                <Text style={styles.statLabel}>Roles</Text>
              </View>
            </View>

            <Text style={styles.sectionHeader}>Filmography</Text>
          </View>
        }
        ListEmptyComponent={
          <Text style={styles.emptyText}>No filmography data yet.</Text>
        }
        contentContainerStyle={styles.listContent}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex:            1,
    backgroundColor: "#000000",
  },
  centered: {
    flex:            1,
    backgroundColor: "#000000",
    alignItems:      "center",
    justifyContent:  "center",
  },
  errorText: {
    color:    "#FF3B30",
    fontSize: 15,
  },
  profileHeader: {
    alignItems: "center",
    paddingTop: 24,
    paddingHorizontal: 24,
    paddingBottom: 16,
  },
  avatar: {
    width:        100,
    height:       100,
    borderRadius: 50,
    marginBottom: 16,
  },
  avatarPlaceholder: {
    backgroundColor: "#1C1C1E",
    alignItems:      "center",
    justifyContent:  "center",
  },
  avatarInitial: {
    color:      "#8E8E93",
    fontSize:   40,
    fontWeight: "700",
  },
  name: {
    color:        "#FFFFFF",
    fontSize:     26,
    fontWeight:   "800",
    textAlign:    "center",
    marginBottom: 8,
  },
  bio: {
    color:      "#AEAEB2",
    fontSize:   14,
    textAlign:  "center",
    lineHeight: 21,
    marginBottom: 16,
  },
  statRow: {
    flexDirection: "row",
    marginBottom:  24,
  },
  stat: {
    alignItems: "center",
    marginHorizontal: 20,
  },
  statValue: {
    color:      "#FFFFFF",
    fontSize:   22,
    fontWeight: "700",
  },
  statLabel: {
    color:    "#8E8E93",
    fontSize: 12,
    marginTop: 2,
  },
  sectionHeader: {
    color:         "#8E8E93",
    fontSize:      13,
    fontWeight:    "600",
    textTransform: "uppercase",
    letterSpacing: 0.5,
    alignSelf:     "flex-start",
    marginBottom:  8,
  },
  listContent: {
    paddingBottom: 40,
  },
  filmRow: {
    flexDirection:   "row",
    alignItems:      "center",
    paddingHorizontal: 24,
    paddingVertical:  12,
    borderBottomWidth: 1,
    borderBottomColor: "#1C1C1E",
  },
  filmInfo: {
    flex: 1,
  },
  charName: {
    color:      "#FFFFFF",
    fontSize:   15,
    fontWeight: "600",
  },
  showName: {
    color:    "#8E8E93",
    fontSize: 13,
    marginTop: 2,
  },
  typeBadge: {
    backgroundColor: "#2C2C2E",
    borderRadius:    6,
    paddingHorizontal: 8,
    paddingVertical:   3,
    marginLeft:      12,
  },
  typeBadgeText: {
    color:    "#AEAEB2",
    fontSize: 11,
  },
  emptyText: {
    color:     "#636366",
    fontSize:  14,
    textAlign: "center",
    marginTop: 24,
  },
});
