# OpenAI Gym Arena

This repository is used as a playground to experiment with OpenAI Gym for the bachelor's thesis.

All of my source files are contained in the `gymkit` module.  

---

To evaluate an algorithm on an environment, use `main.py` as entry point.

First, you need an `Environment` class where you pass the name of the gym environment you want to use to the `name` agrument.
If you need to customize properties of an `Environment`, like a custom `reward` evaluation function, create a subclass which passes the name of its respective gym environment to the superclass' initializer.  
  
  
Next, initialze an `Arena` object with the environment class you wish to evaluate.  
*Important: Don't pass an instance of the class, since `Arena` itself creates instances of `Environment`s when `Agent`s are registered.*  

```python
arena = Arena(mode='training', monitoring_enabled=True)
```  

You can register `Agent`s to an `Arena` where they will be evaluated on the specified environment.  

In this example, we create a `NeatAgent` which uses the 'Neuroevolution of Augmented Topologies' algorithm to evolve a phenotype (a neural network) which will be evaluated on the environment.  

*Note: `elite-size` specifies the number of fittest phenotypes which will have a vote on the decision. E.g. an `elite-size` of 1 would result in a single neural network's output to be used as the decision critieria.*
```python
agent = NeatAgent(elite_size=3)
arena.register(agent)
```
  
  
  
To run the environment and evaluate all registered agents, call the `run` funtion on the arena.
This funtion returns an `Evaluation` object that holds information about the performance of each registered agent.
```python
evaluation = arena.run()
```
