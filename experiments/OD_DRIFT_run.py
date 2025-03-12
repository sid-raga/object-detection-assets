from raga import *
import datetime

def run(project_name, access_key, secret_key, host, name):
    if name == "OD_DRIFT_run":
        run_name = f"FasterRCNN Tests"

        # create test_session object of TestSession instance
        test_session = TestSession(project_name=project_name, run_name= run_name, access_key=access_key, secret_key=secret_key, host=host)


        rules = DriftDetectionRules()
        rules.add(type="drift_detection", dist_metric="Mahalanobis", _class="ALL", threshold=2)

        #To Run OD Test
        edge_case_detection = data_drift_detection(test_session=test_session,
                                                   test_name=f"Drift-Detection-Test-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
                                                   train_dataset_name="bdd_train_1",
                                                   field_dataset_name="bdd_val_1",
                                                   train_embed_col_name="ImageVectorsM1",
                                                   field_embed_col_name = "ImageVectorsM1",
                                                   level = "image",
                                                   rules = rules)


        test_session.add(edge_case_detection)

        test_session.run()
        
if __name__ == "__main__":
    run("ObjectDetection", access_key="BhOkLXzCVX9rs6Y9cGuD",
                           secret_key="SW5JN36ELrMxQzS8GCTRYuSbtEb3s2NMvWKVxPMX", host="https://prod5.ragaai.ai", name="OD_DRIFT_run")