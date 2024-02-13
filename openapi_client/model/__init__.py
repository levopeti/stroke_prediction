# we can not import model classes here because that would create a circular
# reference which would not work in python2
# do not import all trained_models into this module because that uses a lot of memory and stack frames
# if you need the ability to import all trained_models from one package, import them with
# from openapi_client.trained_models import ModelA, ModelB
