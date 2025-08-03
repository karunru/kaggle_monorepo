
from . import cmi_gateway
from .core import templates


class CMIInferenceServer(templates.InferenceServer):
    def _get_gateway_for_test(self, data_paths=None, file_share_dir=None):
        return cmi_gateway.CMIGateway(data_paths)
