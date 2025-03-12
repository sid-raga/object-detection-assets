from raga import *
import datetime

def run(project_name, access_key, secret_key, host, name):
    if name == "OD_DATA_LEAKAGE_test_run":
        run_name = f"FasterRCNN Tests"

        # create test_session object of TestSession instance
        test_session = TestSession(project_name=project_name, run_name= run_name, access_key=access_key, secret_key=secret_key, host=host)

        rules = DLRules()
        rules.add(metric = 'overlapping_samples', metric_threshold = 0.92)


        train_dataset_name = "bdd_train_2"
        field_dataset_name = "bdd_test_1"

        edge_case_detection = data_leakage_test(test_session=test_session,
                                                   test_name=f"Data-Leakage-Test-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
                                                   train_dataset_name=train_dataset_name,
                                                   dataset_name=field_dataset_name,
                                                   type = "data_leakage",
                                                   output_type="image_data",
                                                   train_embed_col_name="ImageVectorsM1",
                                                   embed_col_name = "ImageVectorsM1",
                                                   rules = rules)

        test_session.add(edge_case_detection)

        test_session.run()
        
if __name__ == "__main__":
    run("ObjectDetection", access_key="BhOkLXzCVX9rs6Y9cGuD",
                           secret_key="SW5JN36ELrMxQzS8GCTRYuSbtEb3s2NMvWKVxPMX", host="https://prod5.ragaai.ai", name="OD_DATA_LEAKAGE_test_run")