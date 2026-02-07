# Vision Cascade + Auto-Learning Implementation Plan

## Goal
Complete the automated learning loop:
1. Small model → cascade to big model for uncertain detections
2. Big model success → auto-feed back to replay buffer
3. Review queue for total failures
4. Correction feedback → replay buffer
5. Auto-triggered retraining when buffer fills / system idle

## Track 1: Cascade Logic
- [ ] Add `secondary_model_id` to StreamingConfig
- [ ] Implement cascade in `process_frame()`
- [ ] Add success feedback to replay buffer

## Track 2: Review Service Wiring
- [ ] Wire ReviewService to SQLite image_store
- [ ] Wire submit_review → ReplayBuffer.add_correction()
- [ ] Add correction feedback endpoint

## Track 3: Auto-Training Trigger
- [ ] Buffer threshold trigger
- [ ] Idle detection / scheduled training
- [ ] Training job management

## Track 4: Designer Integration
- [ ] Add Vision section to designer
- [ ] Streaming session UI
- [ ] Review queue UI
- [ ] Training job status

## Track 5: Demo + Tests
- [ ] examples/vision_cascade_demo.py
- [ ] Test cascade flow
- [ ] Test replay buffer
- [ ] Test auto-training trigger

---
Started: 2026-02-06 11:37 CST
