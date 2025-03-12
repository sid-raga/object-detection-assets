from raga import *
import datetime

def run(project_name, access_key, secret_key, host, name):
    if name == "OD_OUTLIER_run":
        dataset_name = "bdd_val_2"
        run_name = f"FasterRCNN Tests"

        # create test_session object of TestSession instance
        test_session = TestSession(project_name=project_name, run_name= run_name, access_key=access_key, secret_key=secret_key, host=host)
        rules = DriftDetectionRules()
        rules.add(type="anomaly_detection", dist_metric="Mahalanobis", _class="ALL", threshold=25)
        edge_case_detection = data_drift_detection(test_session=test_session,
                                                   test_name=f"Outlier_Detection_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
                                                   dataset_name=dataset_name,
                                                   embed_col_name = "ImageVectorsM1",
                                                   output_type = "outlier_detection",
                                                   rules = rules)
        test_session.add(edge_case_detection)

        test_session.run()
        
if __name__ == "__main__":
    run("ObjectDetection", access_key="BhOkLXzCVX9rs6Y9cGuD",
                           secret_key="SW5JN36ELrMxQzS8GCTRYuSbtEb3s2NMvWKVxPMX", host="https://prod5.ragaai.ai", name="OD_OUTLIER_run")