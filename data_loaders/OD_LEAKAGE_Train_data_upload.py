import json
import random
from raga import *
import pandas as pd
import datetime
# from utils.util import get_file


def get_timestamp_x_hours_ago(hours):
    current_time = datetime.datetime.now()
    delta = datetime.timedelta(days=90, hours=hours)
    past_time = current_time - delta
    timestamp = int(past_time.timestamp())
    return timestamp


def img_url(x):
    return StringElement(
        x.replace(
            "s3://raga-engineering",
            "https://raga-engineering.s3.us-east-2.amazonaws.com",
        )
    )


def annotation_v1(row):
    AnnotationsV1 = ImageDetectionObject()
    annotation_json = json.loads(row["AnnotationsV1"].replace("'", '"'))

    if type(annotation_json) == type({}):
        annotation_json = annotation_json["annotations"]

    for idx, detection in enumerate(annotation_json):
        AnnotationsV1.add(
            ObjectDetection(
                Id=idx,
                ClassId=detection["ClassId"],
                ClassName=detection["ClassName"],
                Confidence=float(detection["Confidence"]),
                BBox=detection["BBox"],
                Format="xywh_normalized",
            )
        )
    return AnnotationsV1


def model_inferences(row):
    ModelInferences = ImageDetectionObject()
    model_inference_json = json.loads(row["ModelInferences"].replace("'", '"'))

    if type(model_inference_json) == type({}):
        model_inference_json = model_inference_json["annotations"]

    for idx, detection in enumerate(model_inference_json):
        ModelInferences.add(
            ObjectDetection(
                Id=idx,
                ClassId=detection["ClassId"],
                ClassName=detection["ClassName"],
                Confidence=float(detection["Confidence"]),
                BBox=detection["BBox"],
                Format="xywh_normalized",
            )
        )
    return ModelInferences


def imag_vectors_m1(row):
    ImageVectorsM1 = ImageEmbedding()
    embedding_json = json.loads(row["ImageEmbedding"].replace("'", '"'))

    for embedding in embedding_json:
        ImageVectorsM1.add(Embedding(embedding))

    return ImageVectorsM1


def generate_random():
    classes = ["Yes", "No"]
    return random.choice(classes)


def csv_parser(csv_file):
    df = pd.read_csv(csv_file)
    data_frame = pd.DataFrame()
    data_frame["ImageId"] = df["ImageId"].apply(lambda x: StringElement(x))
    data_frame["ImageUri"] = df["SourceLink"].apply(lambda x: img_url(x))
    data_frame["SourceLink"] = df["SourceLink"].apply(lambda x: img_url(x))
    data_frame["TimeOfCapture"] = df.apply(
        lambda row: TimeStampElement(get_timestamp_x_hours_ago(row.name)), axis=1
    )
    data_frame["Reflection"] = df.apply(
        lambda row: StringElement(random.choice(["Yes", "No"])), axis=1
    )
    data_frame["Overlap"] = df.apply(
        lambda row: StringElement(random.choice(["Yes", "No"])), axis=1
    )
    data_frame["CameraAngle"] = df.apply(
        lambda row: StringElement(random.choice(["Yes", "No"])), axis=1
    )
    data_frame["ModelInferences"] = df.apply(model_inferences, axis=1)
    data_frame["ImageVectorsM1"] = df.apply(imag_vectors_m1, axis=1)
    data_frame["AnnotationsV1"] = df.apply(annotation_v1, axis=1)
    return data_frame

def run(project_name, access_key, secret_key, host, name):

    if name ==  "OD_LEAKAGE_Train_data_upload":
        schema = RagaSchema()
        schema.add("ImageId", PredictionSchemaElement())
        schema.add("ImageUri", ImageUriSchemaElement())
        schema.add("TimeOfCapture", TimeOfCaptureSchemaElement())
        schema.add("SourceLink", FeatureSchemaElement())
        schema.add("Reflection", AttributeSchemaElement())
        schema.add("CameraAngle", AttributeSchemaElement())
        schema.add("Overlap", AttributeSchemaElement())
        schema.add("ImageVectorsM1", ImageEmbeddingSchemaElement(model="FasterRCNN"))
        schema.add("AnnotationsV1", InferenceSchemaElement(model="GT"))
        schema.add("ModelInferences", InferenceSchemaElement(model="FasterRCNN"))
        # create test_session object of TestSession instance
        test_session = TestSession(
            project_name=project_name, access_key=access_key, secret_key=secret_key, host=host
        )
        # get_file(host, test_session.token, "Object_Detection/assets/bdd_with_model_inferences_train_leaks.csv", "Object_Detection/assets/bdd_with_model_inferences_train_leaks.csv")

        pd_data_frame = csv_parser(
            "Object_Detection/assets/bdd_with_model_inferences_train_leaks.csv"
        )
        cred = DatasetCreds(region="us-east-2")

        # create test_ds object of Dataset instance
        test_ds = Dataset(
            test_session=test_session,
            name="BDD_Train_2",
            type=DATASET_TYPE.IMAGE,
            data=pd_data_frame,
            schema=schema,
            creds=cred,
        )

        test_ds.load()

if __name__ == "__main__":
    run("ObjectDetection", access_key="BhOkLXzCVX9rs6Y9cGuD",
                           secret_key="SW5JN36ELrMxQzS8GCTRYuSbtEb3s2NMvWKVxPMX", host="https://prod5.ragaai.ai", name="OD_LEAKAGE_Train_data_upload")