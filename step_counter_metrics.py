import math


class StepCounterErrorMetrics:
    """
    Evaluate predicted step counts against ground-truth step counts.
    """

    def calculate_sample_metrics(self, predicted_steps, ground_truth_steps):
        predicted = int(predicted_steps)
        ground_truth = int(ground_truth_steps)

        signed_error = predicted - ground_truth
        absolute_error = abs(signed_error)

        metrics = {
            "predicted_steps": predicted,
            "ground_truth_steps": ground_truth,
            "signed_error": signed_error,
            "absolute_error": absolute_error,
            "squared_error": signed_error * signed_error,
            "percentage_error": None,
            "absolute_percentage_error": None,
        }

        if ground_truth != 0:
            metrics["percentage_error"] = 100.0 * signed_error / ground_truth
            metrics["absolute_percentage_error"] = 100.0 * absolute_error / ground_truth

        return metrics

    def calculate_summary_metrics(self, predicted_steps, ground_truth_steps):
        predictions = [int(value) for value in predicted_steps]
        ground_truths = [int(value) for value in ground_truth_steps]

        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have the same length.")
        if not predictions:
            raise ValueError("At least one prediction/ground-truth pair is required.")

        sample_metrics = [
            self.calculate_sample_metrics(prediction, ground_truth)
            for prediction, ground_truth in zip(predictions, ground_truths)
        ]

        signed_errors = [item["signed_error"] for item in sample_metrics]
        absolute_errors = [item["absolute_error"] for item in sample_metrics]
        squared_errors = [item["squared_error"] for item in sample_metrics]
        absolute_percentage_errors = [
            item["absolute_percentage_error"]
            for item in sample_metrics
            if item["absolute_percentage_error"] is not None
        ]

        num_samples = len(sample_metrics)
        summary = {
            "num_samples": num_samples,
            "mean_error": sum(signed_errors) / num_samples,
            "mean_absolute_error": sum(absolute_errors) / num_samples,
            "root_mean_squared_error": math.sqrt(sum(squared_errors) / num_samples),
            "mape": None,
            "sample_metrics": sample_metrics,
        }

        if absolute_percentage_errors:
            summary["mape"] = sum(absolute_percentage_errors) / len(absolute_percentage_errors)

        return summary
