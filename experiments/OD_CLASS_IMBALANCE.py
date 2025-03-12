from raga import *
import datetime


rules = ClassImbalanceRules()
rules.add(metric="js_divergence", ideal_distribution="uniform", metric_threshold=0.10, label="ALL")
rules.add(metric="chi_squared_test", ideal_distribution="uniform", metric_threshold=0.10, label="ALL")

run_name = f"FasterRCNN Tests"
print(run_name)
access_key="BhOkLXzCVX9rs6Y9cGuD"
secret_key="SW5JN36ELrMxQzS8GCTRYuSbtEb3s2NMvWKVxPMX"
host="https://prod5.ragaai.ai"

test_session = TestSession(project_name="ObjectDetection", run_name=run_name, access_key=access_key, secret_key=secret_key, host=host)
dataset_name = "bdd_train_1"
distribution_test = class_imbalance_test(test_session=test_session,
                                         dataset_name=dataset_name,
                                         test_name="Class Imbalance Test",
                                         type="class_imbalance",
                                         output_type="object_detection",
                                         annotation_column_name="AnnotationsV1",
                                         rules=rules)

test_session.add(distribution_test)
test_session.run()