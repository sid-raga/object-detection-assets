from raga import *
import datetime

dataset_name = "bdd_val_5"
metric_threshold = 0.72
run_name = "FasterRCNN Tests"

test_session = TestSession(
    project_name="ObjectDetection",
    run_name=run_name,
    access_key="BhOkLXzCVX9rs6Y9cGuD",
    secret_key="SW5JN36ELrMxQzS8GCTRYuSbtEb3s2NMvWKVxPMX",
    host="https://prod5.ragaai.ai"
)

rules = LQRules()
rules.add(metric="mistake_score", label=["ALL"], metric_threshold=metric_threshold)

edge_case_detection = labelling_quality_test(
    test_session=test_session,
    dataset_name=dataset_name,
    test_name="Labeling Quality Test",
    type="labelling_consistency",
    output_type="image_classification",
    mistake_score_col_name="MistakeScore",
    rules=rules,
)
test_session.add(edge_case_detection)
test_session.run()