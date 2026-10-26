import netket as nk

# File defining the sampling rules in the extended Hilbert space


@nk.utils.struct.dataclass
class FixedRule(nk.sampler.rules.MetropolisRule):
    def transition(self, sampler, machine, parameters, state, key, σ):
        return σ, None
