from source.data import char_span_to_token_span
import unittest


class SpanTest(unittest.TestCase):
    def test_from_beginning(self):
        passage = "The College of Arts and Letters was established as the university's first college in 1842 with" \
                  " the first degrees given in 1849. The university's first academic curriculum was modeled after" \
                  " the Jesuit Ratio Studiorum from Saint Louis University. Today the college, housed in " \
                  "O'Shaughnessy Hall, includes 20 departments in the areas of fine arts, humanities, and social" \
                  " sciences, and awards Bachelor of Arts (B.A.) degrees in 33 majors, making it the largest of the" \
                  " university's colleges. There are around 2,500 undergraduates and 750 graduates enrolled in the" \
                  " college."
        start = [0]
        end = [31]
        self.assertEqual([1, 6], char_span_to_token_span(start, end, passage))

    def test_in_the_middle(self):
        passage = "The College of Science was established at the university in 1865 by president Father Patrick " \
                  "Dillon. Dillon's scientific courses were six years of work, including higher-level mathematics" \
                  " courses. Today the college, housed in the newly built Jordan Hall of Science, includes over" \
                  " 1,200 undergraduates in six departments of study – biology, chemistry, mathematics, physics," \
                  " pre-professional studies, and applied and computational mathematics and statistics (ACMS) – " \
                  "each awarding Bachelor of Science (B.S.) degrees. According to university statistics, its science" \
                  " pre-professional program has one of the highest acceptance rates to medical school of any" \
                  " university in the United States."
        start = [78]
        end = [99]
        self.assertEqual([14, 16], char_span_to_token_span(start, end, passage))

