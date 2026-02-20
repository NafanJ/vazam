/**
 * ShowSearchScreen — search and select a show to narrow identification
 *
 * Selecting a show passes it back to HomeScreen via navigation params
 * so the identify call can restrict its search to that cast.
 */

import React, { useCallback, useEffect, useState } from "react";
import {
  ActivityIndicator,
  FlatList,
  SafeAreaView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import { useNavigation } from "@react-navigation/native";
import type { NativeStackNavigationProp } from "@react-navigation/native-stack";

import { listShows, searchShows } from "../api/vazam";
import type { RootStackParamList, Show } from "../types";

type Nav = NativeStackNavigationProp<RootStackParamList, "ShowSearch">;

function ShowRow({ show, onSelect }: { show: Show; onSelect: (s: Show) => void }) {
  return (
    <TouchableOpacity style={styles.row} onPress={() => onSelect(show)}>
      <View style={styles.rowInfo}>
        <Text style={styles.rowTitle} numberOfLines={1}>{show.title}</Text>
        <Text style={styles.rowMeta}>
          {show.media_type}{show.year ? `  ·  ${show.year}` : ""}
        </Text>
      </View>
      <Text style={styles.chevron}>›</Text>
    </TouchableOpacity>
  );
}

export default function ShowSearchScreen(): React.JSX.Element {
  const navigation = useNavigation<Nav>();
  const [query,   setQuery]   = useState("");
  const [shows,   setShows]   = useState<Show[]>([]);
  const [loading, setLoading] = useState(true);

  // Load all shows on mount; switch to search results as user types
  useEffect(() => {
    if (query.trim().length === 0) {
      setLoading(true);
      listShows(100)
        .then(setShows)
        .finally(() => setLoading(false));
    } else {
      setLoading(true);
      const timer = setTimeout(() => {
        searchShows(query.trim())
          .then(setShows)
          .finally(() => setLoading(false));
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [query]);

  const handleSelect = useCallback(
    (show: Show) => {
      // Pass selection back to HomeScreen
      navigation.navigate("Home", { selectedShow: show } as any);
    },
    [navigation]
  );

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.searchBar}>
        <TextInput
          style={styles.input}
          placeholder="Search shows…"
          placeholderTextColor="#636366"
          value={query}
          onChangeText={setQuery}
          autoFocus
          returnKeyType="search"
          clearButtonMode="while-editing"
        />
      </View>

      {loading ? (
        <ActivityIndicator style={styles.spinner} color="#007AFF" />
      ) : shows.length === 0 ? (
        <Text style={styles.emptyText}>
          {query ? `No shows matching "${query}"` : "No shows in database yet."}
        </Text>
      ) : (
        <FlatList
          data={shows}
          keyExtractor={(s) => String(s.id)}
          renderItem={({ item }) => <ShowRow show={item} onSelect={handleSelect} />}
        />
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex:            1,
    backgroundColor: "#000000",
  },
  searchBar: {
    paddingHorizontal: 16,
    paddingVertical:   10,
    borderBottomWidth: 1,
    borderBottomColor: "#1C1C1E",
  },
  input: {
    backgroundColor: "#1C1C1E",
    borderRadius:    10,
    paddingHorizontal: 14,
    paddingVertical:  10,
    color:           "#FFFFFF",
    fontSize:        16,
  },
  spinner: {
    marginTop: 40,
  },
  emptyText: {
    color:     "#636366",
    fontSize:  14,
    textAlign: "center",
    marginTop: 40,
  },
  row: {
    flexDirection:   "row",
    alignItems:      "center",
    paddingHorizontal: 20,
    paddingVertical:  14,
    borderBottomWidth: 1,
    borderBottomColor: "#1C1C1E",
  },
  rowInfo: {
    flex: 1,
  },
  rowTitle: {
    color:      "#FFFFFF",
    fontSize:   16,
    fontWeight: "600",
  },
  rowMeta: {
    color:    "#8E8E93",
    fontSize: 13,
    marginTop: 2,
  },
  chevron: {
    color:    "#3A3A3C",
    fontSize: 22,
    marginLeft: 8,
  },
});
