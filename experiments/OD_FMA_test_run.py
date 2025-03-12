from raga.raga_schema import FMARules
from raga._tests import clustering, failure_mode_analysis
from raga.test_session import TestSession
import datetime

def run(project_name, access_key, secret_key, host, name):
    if name == "OD_FMA_test_run":
        run_name = f"FasterRCNN Tests"

        # create test_session object of TestSession instance
        test_session = TestSession(project_name=project_name, run_name= run_name, access_key=access_key, secret_key=secret_key, host=host)

        rules = FMARules()
        rules.add(metric="Precision", conf_threshold=0.4, metric_threshold=0.60, iou_threshold=0.4, label="ALL")
        rules.add(metric="F1Score", conf_threshold=0.4, metric_threshold=0.60, iou_threshold=0.4, label="ALL")
        rules.add(metric="Recall", conf_threshold=0.4, metric_threshold=0.60, iou_threshold=0.4, label="ALL")

        cls_default = clustering(test_session=test_session, dataset_name = "bdd_val_3", method="k-means", embedding_col="ImageVectorsM1", level="image", args= {"numOfClusters": 6}, force=True)


        edge_case_detection = failure_mode_analysis(test_session=test_session,
                                                    dataset_name = "bdd_val_3",
                                                    test_name = f"Failure_Mode_Analysis_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
                                                    model = "FasterRCNN",
                                                    gt = "GT",
                                                    rules = rules,
                                                    output_type="object_detection",
                                                    type="embedding",
                                                    clustering=cls_default
                                                    )


        test_session.add(edge_case_detection)
        test_session.run()
        
if __name__ == "__main__":
    run("ObjectDetection", access_key="BhOkLXzCVX9rs6Y9cGuD",
                           secret_key="SW5JN36ELrMxQzS8GCTRYuSbtEb3s2NMvWKVxPMX", host="https://prod5.ragaai.ai", name="OD_FMA_test_run")