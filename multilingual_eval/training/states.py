import logging
import math
from torch.utils.data import DataLoader
from dataclasses import dataclass


@dataclass
class TrainingState:
    nb_realignment_samples_expected: int
    nb_realignment_steps_expected: int
    realignment_batch_size: int

    nb_finetuning_samples_expected: int
    nb_finetuning_steps_expected: int
    task_batch_size: int
    accumulation_steps: int
    n_epochs: int

    nb_realignment_samples_seen: int = 0
    nb_realignment_samples_seen_before_restart: int = 0
    nb_realignment_steps_seen: int = 0
    nb_finetuning_steps_seen: int = 0
    has_restarted: bool = False

    def update_from_other_finetuning(self, other: "TrainingState"):
        self.nb_realignment_samples_seen += other.nb_realignment_samples_seen
        self.nb_realignment_samples_seen_before_restart += (
            other.nb_realignment_samples_seen_before_restart
        )
        self.nb_realignment_steps_seen += other.nb_realignment_steps_seen
        self.has_restarted = self.has_restarted or other.has_restarted

    def log_init(self):
        logging.info("*************")
        logging.info("Training info")
        logging.info(f"Number of fine-tuning samples: {self.nb_finetuning_samples_expected}")
        logging.info(f"Number of epochs: {self.n_epochs}")
        logging.info(
            f"Fine-tuning real batch size: {self.task_batch_size * self.accumulation_steps} (batch_size: {self.task_batch_size}, accumulation_steps: {self.accumulation_steps})"
        )
        logging.info(f"Number of total fine-tuning steps: {self.nb_finetuning_steps_expected}")
        if self.nb_realignment_samples_expected > 0:
            logging.info(f"Number of realignment samples: {self.nb_realignment_samples_expected}")
            logging.info(f"Realignment batch size: {self.realignment_batch_size}")
            logging.info(
                f"Number of realignment optimization steps: {self.nb_realignment_steps_expected}"
            )
        else:
            logging.info("No realignment")
        logging.info("*************")

    def log_state(self):
        logging.info(
            f"Finetuning steps done/expected: {self.nb_finetuning_steps_seen} / {self.nb_finetuning_steps_expected}"
        )
        logging.info(
            f"Realignment steps done/expected: {self.nb_realignment_steps_seen} / {self.nb_realignment_steps_expected}"
        )
        logging.info(
            f"Total realignment samples seen: {self.nb_realignment_samples_seen} (has restarted: {self.has_restarted}, distinct: {self.nb_realignment_samples_seen_before_restart})"
        )
        return {
            "finetuning_steps": self.nb_finetuning_steps_seen,
            "realignment_steps": self.nb_realignment_steps_seen,
            "distinct_realignment_samples": self.nb_realignment_samples_seen_before_restart,
            "repeated_realignment_samples": self.nb_realignment_samples_seen,
        }

    @classmethod
    def compute_expected_samples(
        cls,
        strategy: str,
        task_dataset,
        task_dataloader: DataLoader,
        n_epochs: int,
        task_batch_size: int,
        realignment_batch_size: int,
        accumulation_steps=1,
        nb_realignment_steps_before=None,
    ):
        if task_dataset:
            nb_finetuning_steps_expected = (
                math.ceil(len(task_dataloader) / accumulation_steps) * n_epochs
            )
            nb_finetuning_samples_expected = len(task_dataset)
        else:
            nb_finetuning_steps_expected, nb_finetuning_samples_expected = 0, 0

        nb_realignment_steps_expected = 0
        nb_realignment_samples_expected = 0
        if strategy in ["during", 
                        "during_freeze_realign_unfreeze",
                        "during_freeze_realign_unfreeze_last_6",
                        "during_freeze_realign_unfreeze_last_half",
                        "staged"]:
            nb_realignment_steps_expected = nb_finetuning_steps_expected
            nb_realignment_samples_expected = nb_realignment_steps_expected * realignment_batch_size
        elif strategy in ["before", "after", 
                          "freeze_embedding", 
                          "freeze_embedding_pre_realignment", 
                          "freeze_2_encoders_pre_realignment",
                          "freeze_debugging",
                          "freeze_realign_unfreeze", #best one so far
                          "freeze_realign_unfreeze_last_6",
                          "freeze_realign_unfreeze_first_6pls8nd9",
                          "freeze_realign_unfreeze_3to8",
                          "freeze_realign_unfreeze_1to8",
                          "freeze_realign_unfreeze_1to10",
                          "freeze_realign_finetune_1to6",
                          "freeze_realign_finetune_1to8",
                          "freeze_realign_unfreeze_last_half",
                          "freeze_realign_finetune_0to2_9to11"]:
            if nb_realignment_steps_before is not None:
                nb_realignment_steps_expected = nb_realignment_steps_before
                nb_realignment_samples_expected = (
                    nb_realignment_steps_expected * realignment_batch_size
                )
            else:
                nb_realignment_steps_expected = nb_finetuning_steps_expected
                nb_realignment_samples_expected = (
                    nb_realignment_steps_expected * realignment_batch_size
                )
        elif strategy == "before+during":
            nb_realignment_steps_expected = nb_finetuning_steps_expected
            nb_realignment_samples_expected = nb_realignment_steps_expected * realignment_batch_size
            if nb_realignment_steps_before is not None:
                nb_realignment_steps_expected += nb_realignment_steps_before
                nb_realignment_samples_expected += (
                    nb_realignment_steps_expected * realignment_batch_size
                )
            else:
                nb_realignment_steps_expected += nb_finetuning_steps_expected
                nb_realignment_samples_expected += (
                    nb_realignment_steps_expected * realignment_batch_size
                )

        self = cls(
            nb_realignment_samples_expected=nb_realignment_samples_expected,
            nb_realignment_steps_expected=nb_realignment_steps_expected,
            realignment_batch_size=realignment_batch_size,
            nb_finetuning_samples_expected=nb_finetuning_samples_expected,
            nb_finetuning_steps_expected=nb_finetuning_steps_expected,
            task_batch_size=task_batch_size,
            accumulation_steps=accumulation_steps,
            n_epochs=n_epochs,
        )
        self.log_init()

        return self
