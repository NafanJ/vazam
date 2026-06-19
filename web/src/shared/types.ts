// Shared API types for the Vazam web dashboard. Mirror the FastAPI responses
// in api.py (identification results carry character art + show title since
// migration 003).

export type MatchLevel = "confident" | "possible" | "none";

export interface Match {
  match_level: MatchLevel;
  confidence: number;
  character_name?: string | null;
  actor_name?: string | null;
  show_title?: string | null;
  image_url?: string | null;
  window_agreement?: number | null;
  actor_id?: number | null;
  character_id?: number | null;
}

export interface IdentifyResponse {
  results?: Match[];
}

export interface SceneShow {
  title: string;
  speakers_matched: number;
  speakers_total: number;
}

export interface SceneResponse {
  show?: SceneShow | null;
  speakers?: Record<string, Match[]>;
}

export interface Show {
  id: number;
  title: string;
}

export interface Character {
  id: number;
  name: string;
  actor_id?: number | null;
  actor_name?: string | null;
  show_title?: string | null;
  image_url?: string | null;
  occupation?: string | null;
  samples: number;
}

export interface Embedding {
  id: number;
  voice_label?: string | null;
  audio_source?: string | null;
  source_url?: string | null;
  verified?: boolean;
  duration_s?: number | null;
  quality_score?: number | null;
  audio_path?: string | null;
}

export interface CharacterDetail extends Character {
  embeddings?: Embedding[];
}

export interface EnrollResponse {
  samples: number;
}

// One entry in the locally-persisted "previous recordings" list.
export interface HistoryItem {
  character_name?: string | null;
  actor_name?: string | null;
  show_title?: string | null;
  confidence: number;
  match_level: MatchLevel;
  image_url?: string | null;
  ts: number;
}
