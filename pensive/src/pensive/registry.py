# src/pensive/registry.py


def strategy(cls):
    StrategyRegistry.register(cls)
    return cls

def criteria(cls):
    CriteriaRegistry.register(cls)
    return cls

def decision_policy(cls):
    DecisionPolicyRegistry.register(cls)
    return cls



class StrategyRegistry:
    _registry = {}

    @classmethod
    def register(cls, strategy_cls):
        name = strategy_cls.__name__
        cls._registry[name] = strategy_cls
        return strategy_cls

    @classmethod
    def instantiate_all(cls):
        return [strategy_cls() for strategy_cls in cls._registry.values()]


class CriteriaRegistry:
    _registry = {}

    @classmethod
    def register(cls, criteria_cls):
        name = criteria_cls.__name__
        cls._registry[name] = criteria_cls
        return criteria_cls

    @classmethod
    def instantiate_all(cls):
        return [criteria_cls() for criteria_cls in cls._registry.values()]


class DecisionPolicyRegistry:
    _registry = {}

    @classmethod
    def register(cls, policy_cls):
        name = policy_cls.__name__
        cls._registry[name] = policy_cls
        return policy_cls

    @classmethod
    def get(cls, name):
        return cls._registry[name]

    @classmethod
    def instantiate(cls, name):
        return cls._registry[name]()
