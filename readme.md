┌──────────┐
│  Router  │  "Do we need research?"
└────┬─────┘
     ├─→ NO (closed_book) ─→ ┌──────────────┐
     │                        │ Orchestrator │
     └─→ YES ────────────────→│  (planning)  │
         ┌─────────────┐      └──────┬───────┘
         │  Researcher │             │
         │ (Tavily +   │             ↓
         │ synthesizer)│        ┌──────────────┐
         └──────┬──────┘        │   Fanout     │
                └─→ evidence ───→│  (dispatch)  │
                                 └──────┬───────┘
                                        │
                      ┌─────────────────┼─────────────────┐
                      ↓                 ↓                 ↓
                  ┌────────┐      ┌────────┐        ┌────────┐
                  │ Worker │      │ Worker │  ...   │ Worker │
                  │ (sect 1)│      │ (sect 2)│      │ (sect N)│
                  └────┬───┘      └────┬───┘        └────┬───┘
                       └────────────┬──────────────────┘
                                    ↓
                         ┌─────────────────────┐
                         │  merge_content      │
                         │  decide_images      │
                         │  generate_and_place │
                         └─────────────────────┘
                                    ↓
                              ┌──────────┐
                              │  final   │
                              │ markdown │
                              └──────────┘