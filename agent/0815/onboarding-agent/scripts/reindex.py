from server.rag.pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline([("ops-update", "신규 트러블슈팅 답변 본문...")])
    print("Reindexed.")
