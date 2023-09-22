# OLA Project 2023
This is tech GitHub repository for the final project for the Online Learning Applications course, AA 2022/2023.
Project authors: 

- Marco Lucchini ([GitHub](https://github.com/marcolucchini), [Linkedin](https://www.linkedin.com/in/marco-lucchini-294801218/))
- Massimo Perna ([GitHub](https://github.com/maxperna), [Linkedin](https://www.linkedin.com/in/massimo-perna-5ab2b7237/))
- Chiara Mocetti ([GitHub](https://github.com/chiaramocetti), [Linkedin](https://www.linkedin.com/in/chiara-mocetti-757652bb/))
- Giulia Rebay ([GitHub](https://github.com/giuliarebay), [Linkedin](https://www.linkedin.com/in/giuliarebay/))
- Giorgio Romano ([GitHub](https://github.com/grgromano), [Linkedin](https://www.linkedin.com/in/grgromano/))

## Project focus: Pricing and Advertising problems.

The goal of our project was to study solutions to the most typical learning challenges that can be encountered in a pricing and advertising scenario. Specifically, the project explored different learning situations, first tackling simplified versions of both the pricing and advertising problems, in a stationary environment where only one customer profile was considered, then slowly combining simpler tasks to finally tackle dynamic, multiple class learning contexts.
Implementation was fully carried out on Python and heavily relied on an object-oriented approach. We decided to set up opportune Environment, Learner and User classes, each of which were then used in combination with key learning algorithms to find optimal pricing and bidding solutions.

The repository is organised in several branches.
On the main branch, the core implementation is found. This includes:
* All implemented Environments, contained in the folder “Environments”. These interact with the learners and conceptually model the context where the simulations required for the learning to take place happen. In this folder we have also saved the python scripts where the User and Product classes are implemented;
* Learners, defined in the folder “Learners”. Here the core Learner.py class can be found, and all of its extensions used in the following steps;
* Saved plots (which we will provide thorough explanation for in the presentation slides), in the folder “results”;
* The python files used for steps 0 to 3.
Steps 4 to 6, due to their higher complexity, are implemented on separate branches, in which the division of the components of the code are however comparable.