import unittest
from models import truncate_context, answer_question


class TestModels(unittest.TestCase):
    def test_truncate_context(self) -> None:
        """Test the truncate_context function."""
        context = "This is a long context that needs to be truncated."
        question = "What is the context about?"
        truncated = truncate_context(context, question, max_length=10)
        self.assertTrue(len(truncated) < len(context))

    def test_answer_question(self) -> None:
        """Test the answer_question function."""
        context = "The capital of France is Paris."
        question = "What is the capital of France?"
        answer, score = answer_question(context, question)
        self.assertEqual(answer, "Paris")
        self.assertGreater(score, 0)


if __name__ == '__main__':
    unittest.main()
