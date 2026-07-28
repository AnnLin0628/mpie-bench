#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MPIE-Bench manifest Storage layer: single file sqlite,Complete pipeline shared.

table design (Stage 10 Placement, incremental writing in previous stages):
  videos     Video registration(License tiering license_tier: cc0 / restricted)
  shots      Shot segmentation results
  keyframes  candidate/selected keyframes(interaction density score + C0-C3 Candidate ranking)
  persons    Character trajectories and identity clustering(identity_id The only one in Quanyuan)
  refs       Cross-frame reference picture(Four levels of priority tier 1-4 + degree of difference)
  captions   VLM Structured markup
  samples    final training/Evaluation triplet(refs+instruction+target), split Leave fields blank to be split
"""
import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
  video_id TEXT PRIMARY KEY, dataset TEXT, path TEXT,
  license_tier TEXT CHECK(license_tier IN ('cc0','restricted')),
  fps REAL, n_frames INTEGER, meta_json TEXT);
CREATE TABLE IF NOT EXISTS shots (
  shot_id TEXT PRIMARY KEY, video_id TEXT, start_frame INTEGER, end_frame INTEGER);
CREATE TABLE IF NOT EXISTS keyframes (
  kf_id TEXT PRIMARY KEY, video_id TEXT, shot_id TEXT, frame_idx INTEGER,
  n_person INTEGER, density_score REAL, density_level TEXT,  -- C0-C3 candidate prior
  sharpness REAL, frame_path TEXT, selected INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS persons (
  track_id TEXT PRIMARY KEY, video_id TEXT, identity_id TEXT,
  bbox_json TEXT, n_frames INTEGER, face_emb_path TEXT, body_emb_path TEXT);
CREATE TABLE IF NOT EXISTS refs (
  ref_id TEXT PRIMARY KEY, identity_id TEXT, kf_id TEXT,   -- Provenance frame
  tier INTEGER CHECK(tier IN (1,2,3,4)),                    -- 1Across videos 2across shots 3Across aircraft 4Large time distance with same lens
  diversity_score REAL, clean_path TEXT, raw_path TEXT);
CREATE TABLE IF NOT EXISTS captions (
  kf_id TEXT PRIMARY KEY, n_person INTEGER, interaction_type TEXT,
  contact_density_level TEXT, per_person_role TEXT, contact_points TEXT,
  scene_caption TEXT, edit_instruction TEXT, confidence TEXT,
  flag_underage INTEGER DEFAULT 0, raw_json TEXT, needs_review INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS samples (
  sample_id TEXT PRIMARY KEY, kf_id TEXT, video_id TEXT,
  identity_ids TEXT,           -- json list, Used to isolate and segment by identity
  ref_ids TEXT,                -- json list
  instruction TEXT, target_path TEXT,
  density_level TEXT, interaction_type TEXT, n_person INTEGER,
  license_tier TEXT, qc_pass INTEGER DEFAULT 0,
  split TEXT CHECK(split IN ('train','val','test','') OR split IS NULL));
CREATE INDEX IF NOT EXISTS idx_kf_video ON keyframes(video_id);
CREATE INDEX IF NOT EXISTS idx_persons_identity ON persons(identity_id);
CREATE INDEX IF NOT EXISTS idx_samples_split ON samples(split);
"""


def connect(db_path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    conn.row_factory = sqlite3.Row
    return conn


def upsert(conn: sqlite3.Connection, table: str, row: dict):
    keys = ",".join(row)
    ph = ",".join("?" * len(row))
    def _coerce(v):
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False)
        if hasattr(v, "item"):   # numpy If the scalar is directly stored in the database, it will change BLOB, Must transfer Python primitive type
            return v.item()
        return v
    conn.execute(f"INSERT OR REPLACE INTO {table} ({keys}) VALUES ({ph})",
                 [_coerce(v) for v in row.values()])


def rows(conn: sqlite3.Connection, sql: str, args=()) -> list:
    return [dict(r) for r in conn.execute(sql, args).fetchall()]
