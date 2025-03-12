from raga import *
import datetime
from raga._tests import clustering

run_name = "FasterRCNN Tests"

# metadata
# dataset_name = "scenario_imbalance_v1"

#cluster_level
dataset_name = "bdd_train_1"

test_session = TestSession(
    project_name="ObjectDetection",
    run_name=run_name,
    access_key="BhOkLXzCVX9rs6Y9cGuD",
    secret_key="SW5JN36ELrMxQzS8GCTRYuSbtEb3s2NMvWKVxPMX",
    host="https://prod5.ragaai.ai"
)

rules = SBRules()
rules.add(metric="js_divergence", ideal_distribution="uniform", metric_threshold=0.1)
rules.add(metric="chi_squared_test", ideal_distribution="uniform", metric_threshold=0.1)


# clustering is required only at cluster level
cls_default = clustering(test_session=test_session,
                         dataset_name=dataset_name,
                         method="k-means",
                         embedding_col="imageVectorsM1",
                         level="image",
                         args={"numOfClusters": 4}
                         )


# metadata level

# edge_case_detection = scenario_imbalance(test_session=test_session,
#                                             dataset_name = dataset_name,
#                                             test_name = "Scenario_Imbalance_101",
#                                             type = "scenario_imbalance",
#                                             output_type="metadata",
#                                             rules = rules,
#                                             aggregationLevels=["weather"]
#                                              )

# cluster level
# 
edge_case_detection = scenario_imbalance(test_session=test_session,
                                            dataset_name = dataset_name,
                                            test_name = "Scenario_Imbalance",
                                            type = "scenario_imbalance",
                                            output_type="cluster",
                                            embedding= "imageVectorsM1",
                                            rules = rules,
                                            clustering = cls_default
                                             )
test_session.add(edge_case_detection)

test_session.run()