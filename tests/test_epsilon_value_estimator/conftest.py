
from autoverify.verifier.verification_result import CompleteVerificationData

from robustness_experiment_box.database.verification_context import VerificationContext
from robustness_experiment_box.verification_module.verification_module import VerificationModule


class MockVerificationModule(VerificationModule):
    def __init__(self, result_dict: dict):
        self.result_dict = result_dict
        self.name = "MockVerificationModule"

    def verify(self, verification_context: VerificationContext, epsilon: float) -> str | CompleteVerificationData:
        return CompleteVerificationData(self.result_dict[epsilon], took=10.0)
