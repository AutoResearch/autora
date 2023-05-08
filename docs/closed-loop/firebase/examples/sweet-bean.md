# Use SweetBean on Firebase

## Researcher Environment
same as in the stroop example

## Test Subject Environment
Here, we create the main.js on the Firebase server (run python script in your test subject environment folder)
```python
from sweetbean.parameter import TimelineVariable
from sweetbean.sequence import Block, Experiment
from sweetbean.stimulus import TextStimulus, BlankStimulus
from sweetbean.parameter import CodeVariable

color = TimelineVariable("color", ["red", "green"])
word = TimelineVariable("word", ["RED", "GREEN"])

fixation = TextStimulus(duration=800, text='+')
text = TextStimulus(duration=2000, text=word, color=color, choices=["a", "b"])
pause = BlankStimulus(duration=400)

stimulus_sequence = [fixation, text, pause]

block_0 = Block(stimulus_sequence, CodeVariable('condition[0]'))
block_1 = Block(stimulus_sequence, CodeVariable('condition[0]'))
block_2 = Block(stimulus_sequence, CodeVariable('condition[0]'))
experiment = Experiment([block_0, block_1, block_2])

experiment.to_autora(path_package="package.json", path_main='public/design/main.js')

```
