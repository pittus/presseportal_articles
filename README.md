flowchart TD
  A[Textarea-Text] --> B[generate_article(EXPRESS)]
  B --> C[Article_ex]
  A --> D[generate_article(KSTA)]
  D --> E[Article_ks]

  C --> F[judge_article(EXPRESS, Article_ex)]
  E --> G[judge_article(KSTA, Article_ks)]

  F --> H{maybe_revise<br/>(EXPRESS, Article_ex, QC_ex)}
  G --> I{maybe_revise<br/>(KSTA, Article_ks, QC_ks)}

  H -->|optional| J[(Article_ex2?, QC_ex2?)]
  I -->|optional| K[(Article_ks2?, QC_ks2?)]

  J --> L[Render UI + Downloads]
  K --> L
  F --> L
  G --> L
