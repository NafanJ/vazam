/**
 * App.tsx — root navigator
 */

import React from "react";
import { NavigationContainer } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { StatusBar } from "react-native";

import HomeScreen          from "./src/screens/HomeScreen";
import ResultsScreen       from "./src/screens/ResultsScreen";
import ActorProfileScreen  from "./src/screens/ActorProfileScreen";
import ShowSearchScreen    from "./src/screens/ShowSearchScreen";
import type { RootStackParamList } from "./src/types";

const Stack = createNativeStackNavigator<RootStackParamList>();

const SCREEN_OPTIONS = {
  headerStyle:      { backgroundColor: "#000000" },
  headerTintColor:  "#FFFFFF",
  headerTitleStyle: { fontWeight: "700" as const },
  contentStyle:     { backgroundColor: "#000000" },
} as const;

export default function App(): React.JSX.Element {
  return (
    <NavigationContainer>
      <StatusBar barStyle="light-content" backgroundColor="#000000" />
      <Stack.Navigator screenOptions={SCREEN_OPTIONS}>
        <Stack.Screen
          name="Home"
          component={HomeScreen}
          options={{ headerShown: false }}
        />
        <Stack.Screen
          name="Results"
          component={ResultsScreen}
          options={{ title: "Results", headerBackTitle: "Listen" }}
        />
        <Stack.Screen
          name="ActorProfile"
          component={ActorProfileScreen}
          options={({ route }) => ({
            title: route.params.actorName,
            headerBackTitle: "Results",
          })}
        />
        <Stack.Screen
          name="ShowSearch"
          component={ShowSearchScreen}
          options={{ title: "Select a Show", presentation: "modal" }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
