# openai-gym-playground

This repository is used as a playground to experiment with OpenAI Gym for the bachelor's thesis.

To evaluate an algorithm on an environment, use `main.py` as entry point.

First, create a `Config` object and pass the name of the gym environment you want to use to the `name` agrument.
```python
config = Config(name='Pendulum-v0')
```


Then, create an `Arena` object. You can register `Agent`s to an `Arena` where they will be evaluated on the environment specified in the `Config` object.
In this example, we create a `NeatAgent` which uses the 'Neuroevolution of Augmented Topologies' algorithm to evolve a phenotype (a neural network) which will be evaluated on the environment.
```python
arena = Arena(config)

agent = NeatAgent(elite_size=3)
arena.register(agent)
```


To run the environment and evaluate all registered agents, call the `run` funtion on the arena.
This funtion returns an `Evaluation` object, which holds information about the performance of each registered agent.
```python
evaluations = arena.run()
```
