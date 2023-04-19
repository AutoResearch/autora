class EligibilityOptions:
    @staticmethod
    def age(minimum: int, maximum: int):
        return {
            "_cls": "web.eligibility.models.AgeRangeEligibilityRequirement",
            "attributes": [
                {
                    "type": "min",
                    "max": 100,
                    "min": 18,
                    "value": minimum,
                    "name": "min_age",
                    "label": "Minimum Age",
                    "default_value": None,
                    "unit": "",
                    "step": 1,
                },
                {
                    "type": "max",
                    "max": 100,
                    "min": 18,
                    "value": maximum,
                    "name": "max_age",
                    "label": "Maximum Age",
                    "default_value": None,
                    "unit": "",
                    "step": 1,
                },
            ],
            "query": {"id": "54ac6ea9fdf99b2204feb893"},
        }

    @staticmethod
    def nationality(nationality: str, index: int):
        return {
            "_cls": "web.eligibility.models.SelectAnswerEligibilityRequirement",
            "attributes": [
                {
                    "label": nationality,
                    "name": nationality,
                    "value": True,
                    "index": index,
                },
            ],
            "query": {"id": "54bef0fafdf99b15608c504e"},
        }

    @staticmethod
    def vision():
        return {
            "_cls": "web.eligibility.models.SelectAnswerEligibilityRequirement",
            "attributes": [{"label": "Yes", "name": "Yes", "value": True, "index": 0}],
            "query": {"id": "57a0c4d2717b34954e81b919"},
        }

    @staticmethod
    def first_language(language: str):
        return {
            "_cls": "web.eligibility.models.SelectAnswerEligibilityRequirement",
            "attributes": [
                {"label": language, "name": language, "value": True, "index": 18}
            ],
            "query": {"id": "54ac6ea9fdf99b2204feb899"},
        }

    @staticmethod
    def previous_studies(studies):
        return {
            "_cls": "web.eligibility.models.PreviousStudiesEligibilityRequirement",
            "attributes": [
                {
                    "label": study["name"],
                    "value": True,
                    "id": study["id"],
                }
                for study in studies
            ],
            "query": {"id": ""},
        }
