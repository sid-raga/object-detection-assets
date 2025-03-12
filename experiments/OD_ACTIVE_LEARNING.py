from raga import *
import datetime

dataset_name = "bdd_train_1"
budget = 100
run_name = "FasterRCNN Tests"

test_session = TestSession(
    project_name="ObjectDetection",
    run_name=run_name,
    access_key="BhOkLXzCVX9rs6Y9cGuD",
    secret_key="SW5JN36ELrMxQzS8GCTRYuSbtEb3s2NMvWKVxPMX",
    host="https://prod5.ragaai.ai"
)

edge_case_detection = active_learning(test_session=test_session,
                                      dataset_name = dataset_name,
                                      test_name = "Active Learning Test",
                                      type = "active_learning",
                                      output_type="curated_dataset",
                                      embed_col_name="ImageVectorsM1",
                                      budget=budget)

test_session.add(edge_case_detection)

test_session.run()