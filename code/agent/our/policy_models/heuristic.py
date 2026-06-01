"""This class uses a simple heuristic to select actions for the agent.

Good agents:
 - propose parties with the highest probability of being good players
 - vote no on parties that contain likely-evil players
 - always pass quests

Evil agents (Minions):
 - propose parties that include exactly one evil player (preferring the one who
   currently looks cleanest) plus the most-trusted-looking good players for cover
 - approve parties that contain an evil player; reject "too clean" parties only
   when it is safe to do so; never block so hard that cover is blown
 - fail quests strategically (smart/cautious): build trust on Quest 1, coordinate
   so only one evil fails a shared quest, and fail aggressively late or when behind
"""

from enum import Enum
from agent_base import ATEAM
import random

# class ATEAM(Enum):
#     GOOD = 1
#     EVIL = 2

# Tunables for evil play
PASS_FIRST_QUEST = True       # if True, evil passes Quest 1 to build trust (party size is only 2)
EVIL_BURNED_THRESHOLD = 0.7   # marks an evil as exposed; proposal still includes the least-exposed evil


class HeuristicOracle(object):
    def __init__(self, role):
        self.role = role    # good is 1, evil is 2. This will be of type ATEAM

        # Stable identity (set after private data arrives, via set_identity)
        self.self_name = None
        self.evil_teammates = set()   # lowercase names of known evil teammates (excluding self)

        # Volatile game context (set each turn via update_context)
        self.failed_party_votes = 0
        self.quest_number = 1
        self.good_wins = 0
        self.evil_wins = 0

        # Last party seen during vote_for_party, used to coordinate quest fails
        self._last_party = []

    def set_identity(self, self_name, evil_teammates):
        """Inject the agent's own name and the names of its known evil teammates.

        Called once after private data arrives (the oracle is constructed before
        private data exists, so identity cannot come through the constructor)."""
        self.self_name = (self_name or "").lower()
        self.evil_teammates = {n.lower() for n in (evil_teammates or set())}

    def update_context(self, failed_party_votes, quest_number, good_wins, evil_wins):
        """Inject the volatile game-pressure signals before an action is selected."""
        self.failed_party_votes = failed_party_votes
        self.quest_number = quest_number
        self.good_wins = good_wins
        self.evil_wins = evil_wins

    def _evil_set(self):
        """Names (lowercase) of all known evil players: self plus known teammates."""
        s = set(self.evil_teammates)
        if self.self_name:
            s.add(self.self_name)
        return s

    def _is_evil_name(self, name):
        return name.lower() in self._evil_set()

    def get_action(self, state):
        """place holder for action selection funciton. Right now this does nothing"""
        return {}

    def opinion_on_party(self, party, probabilities):
        """"Do not accept parties with evil players"""
        for memeber in party:
            if probabilities[memeber]['evil'] > probabilities[memeber]['good']:
                return False
        return True


    def vote_for_party(self, party, probabilities):
        """Decide whether to approve a proposed party."""
        # Record the party so vote_for_quest can coordinate which evil fails.
        self._last_party = list(party)

        if self.role == ATEAM.EVIL:
            return self._evil_vote_for_party(party, probabilities)

        # GOOD behavior (unchanged):
        if len(party) == 2: # always agree to the first party
            return True
        if probabilities == None:
            raise ValueError("Probabilities are None")
        for memeber in party: # TODO change this so that these are the top k players in the party
            # add a threshold of confidence: if you think the guy is more than 65% evil, then vote no
            if (probabilities[memeber]['evil'] - probabilities[memeber]['good']) > 0:
                return False
        return True

    def _evil_vote_for_party(self, party, probabilities):
        """Evil party vote: approve teams with an evil; reject too-clean teams only when safe."""
        n_evil = sum(1 for m in party if self._is_evil_name(m))
        # Team is "too clean" (all good) when evil is one fail from winning:
        # reject unless agent_acl's final-proposal override forces approval at >=4.
        if n_evil == 0 and self.evil_wins == 2:
            return False
        # Near the rejection limit: approve so we don't look like the lone blocker
        # (agent_acl also force-approves at >=4, but back off earlier for cover).
        if self.failed_party_votes >= 3:
            return True
        # Team contains an evil who can fail the quest -> approve (incl. 2-evil stacks).
        if n_evil >= 1:
            return True
        # Team is "too clean" (all good):
        if self.failed_party_votes <= 1 and self.quest_number >= 2:
            return False  # early in the proposal cycle, safe to reject a clean team
        return True  # otherwise preserve cover

    def propose_party(self, party_size, probabilities):
        """Propose a party. Good: top-good players. Evil: one evil + trusted goods."""
        if self.role == ATEAM.EVIL:
            return self._evil_propose_party(party_size, probabilities)

        # GOOD behavior (unchanged):
        players_in_oreder = self.order_players(probabilities)
        if party_size ==2:
            return [players_in_oreder[0], random.choice(players_in_oreder[1:])]
        return players_in_oreder[:party_size]

    def _evil_propose_party(self, party_size, probabilities):
        """Evil proposal: include at most one evil (the cleanest-looking) plus trusted goods."""
        order_good = self.order_players(probabilities)  # high P(good) first = how OTHERS see them
        evils = [n for n in order_good if self._is_evil_name(n)]
        goods = [n for n in order_good if not self._is_evil_name(n)]

        party = []
        if evils:
            # Prefer the evil who currently looks cleanest (lowest P(evil)).
            # EVIL_BURNED_THRESHOLD documents when an evil is likely exposed, but
            # Phase 1 policy still requires proposals to include a known evil.
            cand = min(evils, key=lambda n: probabilities[n]['evil'])
            party.append(cand)
        # Fill remaining slots with the most-trusted-looking goods (best cover).
        for n in goods:
            if len(party) >= party_size:
                break
            party.append(n)
        # Final backfill if still short (e.g. the only evil was burned and skipped).
        for n in order_good:
            if len(party) >= party_size:
                break
            if n not in party:
                party.append(n)
        return party[:party_size]

    def vote_for_quest(self):
        """Decide whether to pass (True) or fail (False) a quest."""
        if self.role != ATEAM.EVIL:
            return True  # good always passes

        # Coordinate so only one evil fails a shared quest (a quest needs only one fail).
        evils_on_quest = [m for m in self._last_party if self._is_evil_name(m)]
        if len(evils_on_quest) >= 2:
            designated = min(evils_on_quest)  # deterministic: both teammates compute the same name
            if self.self_name != designated:
                return True  # let the teammate fail; I pass to spread blame

        # Whether to fail at all:
        if self.evil_wins >= 2:
            return False  # failing this quest wins the game
        if self.quest_number == 1 and PASS_FIRST_QUEST:
            return True  # build trust on the size-2 first quest; don't expose a lone evil early
        if self.good_wins == 2:
            return False  # must fail now or good wins next quest
        return False  # mid-game default: fail the quest we're on

    def chose_assassin_target(self, probabilities):
        """Defensive stub; not reached in the 6p no-special-role config (both evils are Minions)."""
        return self.order_players(probabilities)[0]

    def order_players(self, probabilities):
        """Order players by the probability of being good"""
        return sorted(probabilities.keys(), key=lambda x: probabilities[x]['good'], reverse=True)
