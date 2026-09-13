from travel_agent.i18n import language_instruction,t

def test_hungarian_labels_are_available():
    assert t("weather","hu")=="Időjárás" and "Hungarian" in language_instruction("hu")
