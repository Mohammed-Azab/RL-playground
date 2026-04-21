"""
callbacks.py — Custom SB3 callbacks for TensorBoard logging.
"""

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from icm import ICMWrapper


class ICMLoggingCallback(BaseCallback):
    """Logs ICM-specific metrics into the SB3 TensorBoard writer.

    Expects the training env stack to be: DummyVecEnv → Monitor → ICMWrapper → base env.
    Reads stats stored on the ICMWrapper instance each step and records them
    via self.logger so they appear alongside standard SB3 metrics in TensorBoard.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._icm_env: ICMWrapper | None = None

    def _init_callback(self) -> None:
        # Unwrap DummyVecEnv → Monitor → ICMWrapper
        inner = self.model.env.envs[0]
        if isinstance(inner, Monitor):
            inner = inner.env
        if not isinstance(inner, ICMWrapper):
            raise RuntimeError(
                "ICMLoggingCallback expects the env stack to contain an ICMWrapper, "
                f"but found {type(inner).__name__}."
            )
        self._icm_env = inner

    def _on_step(self) -> bool:
        env = self._icm_env
        self.logger.record("icm/intrinsic_reward", env.last_intrinsic_reward)
        self.logger.record("icm/forward_loss", env.last_forward_loss)
        self.logger.record("icm/inverse_loss", env.last_inverse_loss)
        self.logger.record("icm/total_loss", env.last_icm_loss)
        self.logger.record("icm/ep_intrinsic_reward", env.last_ep_intrinsic_reward)
        return True
