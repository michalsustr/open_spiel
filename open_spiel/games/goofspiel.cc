// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/goofspiel.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace goofspiel {
namespace {

const GameType kGameType{
    /*short_name=*/"goofspiel",
    /*long_name=*/"Goofspiel",
    GameType::Dynamics::kSimultaneous,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/10,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{
      {"imp_info", GameParameter(kDefaultImpInfo)},
      {"num_cards", GameParameter(kDefaultNumCards)},
      {"players", GameParameter(kDefaultNumPlayers)},
      {"points_order", GameParameter(
          static_cast<std::string>(kDefaultPointsOrder))},
      {"returns_type",GameParameter(
          static_cast<std::string>(kDefaultReturnsType))}
    },
    /*default_loadable=*/true,
    /*provides_factored_observation_string=*/true
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new GoofspielGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

PointsOrder ParsePointsOrder(const std::string& po_str) {
  if (po_str == "random") {
    return PointsOrder::kRandom;
  } else if (po_str == "descending") {
    return PointsOrder::kDescending;
  } else if (po_str == "ascending") {
    return PointsOrder::kAscending;
  } else {
    SpielFatalError(
        absl::StrCat("Unrecognized points_order parameter: ", po_str));
  }
}

ReturnsType ParseReturnsType(const std::string& returns_type_str) {
  if (returns_type_str == "win_loss") {
    return ReturnsType::kWinLoss;
  } else if (returns_type_str == "point_difference") {
    return ReturnsType::kPointDifference;
  } else if (returns_type_str == "total_points") {
    return ReturnsType::kTotalPoints;
  } else {
    SpielFatalError(absl::StrCat("Unrecognized returns_type parameter: ",
                                 returns_type_str));
  }
}

}  // namespace


class GoofspielObserver : public Observer {
 public:
  explicit GoofspielObserver(IIGObservationType iig_obs_type)
      : Observer(/*has_string=*/true, /*has_tensor=*/true),
        iig_obs_type_(iig_obs_type) {}

  void WriteTensor(const State& observed_state, int player,
                   Allocator* allocator) const override {
    const GoofspielState& state =
        open_spiel::down_cast<const GoofspielState&>(observed_state);
    const GoofspielGame& game =
        open_spiel::down_cast<const GoofspielGame&>(*state.GetGame());
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, game.NumPlayers());

    if (iig_obs_type_.public_info) {
      // Point totals: one-hot vector encoding points, per player.
      auto out = allocator->Get("point_totals",
                                {game.NumPlayers(), game.MaxPointSlots()});
      Player p = player;
      for (int n = 0; n < game.NumPlayers(); state.NextPlayer(&n, &p)) {
        out.at(n, state.points_[p]) = 1.0;
      }

      if (!game.IsImpInfo()) {
        // Bit vectors encoding all players' hands.
        auto out = allocator->Get("player_hands",
                                  {game.NumPlayers(), game.NumCards()});
        Player p = player;
        for (int n = 0; n < game.NumPlayers(); state.NextPlayer(&n, &p)) {
          for (int c = 0; c < game.NumCards(); ++c) {
            out.at(n, c) = state.player_hands_[p][c];
          }
        }
      }

      {  // Sequence of who won each trick.
        auto out = allocator->Get("win_sequence",
                                  {game.NumRounds(), game.NumPlayers()});
        for (int i = 0; i < state.win_sequence_.size(); ++i) {
          if (state.win_sequence_[i] != kInvalidPlayer)
            out.at(i, state.win_sequence_[i]) = 1.0;
        }
      }

      if (iig_obs_type_.perfect_recall) {
        {  // Point card sequence.
          auto out = allocator->Get("point_card_sequence",
                                    {game.NumRounds(), game.NumCards()});
          for (int round = 0; round < state.point_card_sequence_.size();
               ++round) {
            out.at(round, state.point_card_sequence_[round]) = 1.0;
          }
        }
      } else {
        {  // Current point card.
          auto out = allocator->Get("point_card", {game.NumCards()});
          if (!state.point_card_sequence_.empty())
            out.at(state.point_card_sequence_.back()) = 1.0;
        }
      }
    }

    if (game.IsImpInfo()
        && iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
      {  // Bit vector of observing player's hand.
        auto out = allocator->Get("player_hand", {game.NumCards()});
        for (int c = 0; c < game.NumCards(); ++c) {
          out.at(c) = state.player_hands_[player][c];
        }
      }

      if (iig_obs_type_.perfect_recall) {
        // The observing player's action sequence.
        auto out = allocator->Get("player_action_sequence",
                                  {game.NumRounds(), game.NumCards()});
        for (int round = 0; round < state.actions_history_.size(); ++round) {
          out.at(round, state.actions_history_[round][player]) = 1.0;
        }
      }
    }
  }

  std::string StringFrom(const State& observed_state,
                         int player) const override {
    const GoofspielState& state =
        open_spiel::down_cast<const GoofspielState&>(observed_state);
    const GoofspielGame& game =
        open_spiel::down_cast<const GoofspielGame&>(*state.GetGame());
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, game.NumPlayers());
    std::string result;

    if (game.IsImpInfo()
        && iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
      // Only my hand
      absl::StrAppend(&result, "P", player, " hand: ");
      for (int c = 0; c < game.NumCards(); ++c) {
        if (state.player_hands_[player][c])
          absl::StrAppend(&result, c + 1, " ");
      }
      absl::StrAppend(&result, "\n");

      if (iig_obs_type_.perfect_recall) {
        // Also show the player's sequence. We need this to ensure perfect
        // recall because two betting sequences can lead to the same hand and
        // outcomes if the opponent chooses differently.
        absl::StrAppend(&result, "P", player, " action sequence: ");
        for (int i = 0; i < state.actions_history_.size(); ++i) {
          absl::StrAppend(&result, state.actions_history_[i][player], " ");
        }
        absl::StrAppend(&result, "\n");
      }
    }

    if (iig_obs_type_.public_info) {
      if (iig_obs_type_.perfect_recall) {
        absl::StrAppend(&result, "Point card sequence: ");
        for (int i = 0; i < state.point_card_sequence_.size(); ++i) {
          absl::StrAppend(&result, 1 + state.point_card_sequence_[i], " ");
        }
        absl::StrAppend(&result, "\n");
      } else {
        absl::StrAppend(&result, "Current point card: ",
                        state.CurrentPointValue(), "\n");
      }

      if (!game.IsImpInfo()) {
        // Show the hands in the perfect info case.
        for (auto p = Player{0}; p < game.NumPlayers(); ++p) {
          absl::StrAppend(&result, "P", p, " hand: ");
          for (int c = 0; c < game.NumCards(); ++c) {
            if (state.player_hands_[p][c])
              absl::StrAppend(&result, c + 1, " ");
          }
          absl::StrAppend(&result, "\n");
        }
      }

      absl::StrAppend(&result, "Win sequence: ");
      for (int i = 0; i < state.win_sequence_.size(); ++i) {
        absl::StrAppend(&result, state.win_sequence_[i], " ");
      }
      absl::StrAppend(&result, "\n");

      absl::StrAppend(&result, "Points: ");
      for (auto p = Player{0}; p < game.NumPlayers(); ++p) {
        absl::StrAppend(&result, state.points_[p], " ");
      }
      absl::StrAppend(&result, "\n");
    }
    return result;
  }

 private:
  IIGObservationType iig_obs_type_;
};

GoofspielState::GoofspielState(std::shared_ptr<const Game> game, int num_cards,
                               PointsOrder points_order, bool impinfo,
                               ReturnsType returns_type)
    : SimMoveState(game),
      num_cards_(num_cards),
      points_order_(points_order),
      returns_type_(returns_type),
      impinfo_(impinfo),
      current_player_(kInvalidPlayer),
      winners_({}),
      turns_(0),
      point_card_(-1),
      point_card_sequence_({}),
      win_sequence_({}),
      actions_history_({}) {
  // Points and point-card deck.
  points_.resize(num_players_);
  std::fill(points_.begin(), points_.end(), 0);

  // Player hands.
  player_hands_.clear();
  for (auto p = Player{0}; p < num_players_; ++p) {
    std::vector<bool> hand(num_cards_, true);
    player_hands_.push_back(hand);
  }

  // Set the points card index.
  if (points_order_ == PointsOrder::kRandom) {
    point_card_ = -1;
    current_player_ = kChancePlayerId;
  } else if (points_order_ == PointsOrder::kAscending) {
    DealPointCard(0);
    current_player_ = kSimultaneousPlayerId;
  } else if (points_order_ == PointsOrder::kDescending) {
    DealPointCard(num_cards - 1);
    current_player_ = kSimultaneousPlayerId;
  }
}

int GoofspielState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else {
    return current_player_;
  }
}

void GoofspielState::DealPointCard(int point_card) {
  SPIEL_CHECK_GE(point_card, 0);
  SPIEL_CHECK_LT(point_card, num_cards_);
  point_card_ = point_card;
  point_card_sequence_.push_back(point_card);
}

void GoofspielState::DoApplyAction(Action action_id) {
  if (IsSimultaneousNode()) {
    ApplyFlatJointAction(action_id);
    return;
  }
  SPIEL_CHECK_TRUE(IsChanceNode());
  DealPointCard(action_id);
  current_player_ = kSimultaneousPlayerId;
}

void GoofspielState::DoApplyActions(const std::vector<Action>& actions) {
  // Check the actions are valid.
  SPIEL_CHECK_EQ(actions.size(), num_players_);
  for (auto p = Player{0}; p < num_players_; ++p) {
    const int action = actions[p];
    SPIEL_CHECK_GE(action, 0);
    SPIEL_CHECK_LT(action, num_cards_);
    SPIEL_CHECK_TRUE(player_hands_[p][action]);
  }

  // Find the highest bid
  int max_bid = -1;
  int num_max_bids = 0;
  int max_bidder = -1;

  for (int p = 0; p < actions.size(); ++p) {
    if (actions[p] > max_bid) {
      max_bid = actions[p];
      num_max_bids = 1;
      max_bidder = p;
    } else if (actions[p] == max_bid) {
      num_max_bids++;
    }
  }

  if (num_max_bids == 1) {
    // Winner takes the point card.
    points_[max_bidder] += CurrentPointValue();
    win_sequence_.push_back(max_bidder);
  } else {
    // Tied among several players: discarded.
    win_sequence_.push_back(kInvalidPlayer);
  }

  // Add these actions to the history.
  actions_history_.push_back(actions);

  // Remove the cards from the player's hands.
  for (auto p = Player{0}; p < num_players_; ++p) {
    player_hands_[p][actions[p]] = false;
  }

  // Deal the next point card.
  if (points_order_ == PointsOrder::kRandom) {
    current_player_ = kChancePlayerId;
    point_card_ = -1;
  } else if (points_order_ == PointsOrder::kAscending) {
    if (point_card_ < num_cards_ - 1) DealPointCard(point_card_ + 1);
  } else if (points_order_ == PointsOrder::kDescending) {
    if (point_card_ > 0) DealPointCard(point_card_ - 1);
  }

  // Next player's turn.
  turns_++;

  // No choice at the last turn, so we can play it now
  // We use DoApplyAction(s) not to modify the history, as these actions are
  // not available in the tree.
  if (turns_ == num_cards_ - 1) {
    // There might be a chance event
    if (IsChanceNode()) {
      auto legal_actions = LegalChanceOutcomes();
      SPIEL_CHECK_EQ(legal_actions.size(), 1);
      DoApplyAction(legal_actions.front());
    }

    // Each player plays their last card
    std::vector<Action> actions(num_players_);
    for (auto p = Player{0}; p < num_players_; ++p) {
      auto legal_actions = LegalActions(p);
      SPIEL_CHECK_EQ(legal_actions.size(), 1);
      actions[p] = legal_actions[0];
    }
    DoApplyActions(actions);
  } else if (turns_ == num_cards_) {
    // Game over - determine winner.
    int max_points = -1;
    for (auto p = Player{0}; p < num_players_; ++p) {
      if (points_[p] > max_points) {
        winners_.clear();
        max_points = points_[p];
        winners_.insert(p);
      } else if (points_[p] == max_points) {
        winners_.insert(p);
      }
    }
  }
}

std::vector<std::pair<Action, double>> GoofspielState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  std::set<int> played(point_card_sequence_.begin(),
                       point_card_sequence_.end());
  std::vector<std::pair<Action, double>> outcomes;
  const int n = num_cards_ - played.size();
  const double p = 1.0 / n;
  outcomes.reserve(n);
  for (int i = 0; i < num_cards_; ++i) {
    if (played.count(i) == 0) outcomes.emplace_back(i, p);
  }
  SPIEL_CHECK_EQ(outcomes.size(), n);
  return outcomes;
}

std::vector<Action> GoofspielState::LegalActions(Player player) const {
  if (player == kSimultaneousPlayerId) return LegalFlatJointActions();
  if (player == kChancePlayerId) return LegalChanceOutcomes();
  if (player == kTerminalPlayerId) return std::vector<Action>();
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::vector<Action> movelist;
  for (int bid = 0; bid < player_hands_[player].size(); ++bid) {
    if (player_hands_[player][bid]) {
      movelist.push_back(bid);
    }
  }
  return movelist;
}

std::string GoofspielState::ActionToString(Player player,
                                           Action action_id) const {
  if (player == kSimultaneousPlayerId)
    return FlatJointActionToString(action_id);
  SPIEL_CHECK_GE(action_id, 0);
  SPIEL_CHECK_LT(action_id, num_cards_);
  if (player == kChancePlayerId) {
    return absl::StrCat("Deal ", action_id + 1);
  } else {
    return absl::StrCat("[P", player, "]Bid: ", (action_id + 1));
  }
}

std::string GoofspielState::ToString() const {
  std::string points_line = "Points: ";
  std::string result = "";

  for (auto p = Player{0}; p < num_players_; ++p) {
    absl::StrAppend(&points_line, points_[p]);
    absl::StrAppend(&points_line, " ");
    absl::StrAppend(&result, "P");
    absl::StrAppend(&result, p);
    absl::StrAppend(&result, " hand: ");
    for (int c = 0; c < num_cards_; ++c) {
      if (player_hands_[p][c]) {
        absl::StrAppend(&result, c + 1);
        absl::StrAppend(&result, " ");
      }
    }
    absl::StrAppend(&result, "\n");
  }

  // In imperfect information, the full state depends on both betting sequences
  if (impinfo_) {
    for (auto p = Player{0}; p < num_players_; ++p) {
      absl::StrAppend(&result, "P", p, " actions: ");
      for (int i = 0; i < actions_history_.size(); ++i) {
        absl::StrAppend(&result, actions_history_[i][p]);
        absl::StrAppend(&result, " ");
      }
      absl::StrAppend(&result, "\n");
    }
  }

  absl::StrAppend(&result, "Point card sequence: ");
  for (int i = 0; i < point_card_sequence_.size(); ++i) {
    absl::StrAppend(&result, 1 + point_card_sequence_[i], " ");
  }
  absl::StrAppend(&result, "\n");

  return result + points_line + "\n";
}

bool GoofspielState::IsTerminal() const { return (turns_ == num_cards_); }

std::vector<double> GoofspielState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(num_players_, 0.0);
  }

  if (returns_type_ == ReturnsType::kWinLoss) {
    if (winners_.size() == num_players_) {
      // All players have same number of points? This is a draw.
      return std::vector<double>(num_players_, 0.0);
    } else {
      int num_winners = winners_.size();
      int num_losers = num_players_ - num_winners;
      std::vector<double> returns(num_players_, (-1.0 / num_losers));
      for (const auto& winner : winners_) {
        returns[winner] = 1.0 / num_winners;
      }
      return returns;
    }
  } else if (returns_type_ == ReturnsType::kPointDifference) {
    std::vector<double> returns(num_players_, 0);
    double sum = 0;
    for (Player p = 0; p < num_players_; ++p) {
      returns[p] = points_[p];
      sum += points_[p];
    }
    for (Player p = 0; p < num_players_; ++p) {
      returns[p] -= sum / num_players_;
    }
    return returns;
  } else if (returns_type_ == ReturnsType::kTotalPoints) {
    std::vector<double> returns(num_players_, 0);
    for (Player p = 0; p < num_players_; ++p) {
      returns[p] = points_[p];
    }
    return returns;
  } else {
    SpielFatalError(absl::StrCat("Unrecognized returns type: ", returns_type_));
  }
}

std::string GoofspielState::InformationStateString(Player player) const {
  const GoofspielGame& game =
      open_spiel::down_cast<const GoofspielGame&>(*game_);
  return game.info_state_observer_->StringFrom(*this, player);
}

std::string GoofspielState::ObservationString(Player player) const {
  const GoofspielGame& game =
      open_spiel::down_cast<const GoofspielGame&>(*game_);
  return game.default_observer_->StringFrom(*this, player);
}

void GoofspielState::NextPlayer(int* count, Player* player) const {
  *count += 1;
  *player = (*player + 1) % num_players_;
}

void GoofspielState::InformationStateTensor(Player player,
                                            absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const GoofspielGame& game =
      open_spiel::down_cast<const GoofspielGame&>(*game_);
  game.info_state_observer_->WriteTensor(*this, player, &allocator);
}

void GoofspielState::ObservationTensor(Player player,
                                       absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const GoofspielGame& game =
      open_spiel::down_cast<const GoofspielGame&>(*game_);
  game.default_observer_->WriteTensor(*this, player, &allocator);
}

std::unique_ptr<State> GoofspielState::Clone() const {
  return std::unique_ptr<State>(new GoofspielState(*this));
}

GoofspielGame::GoofspielGame(const GameParameters& params)
    : Game(kGameType, params),
      num_cards_(ParameterValue<int>("num_cards")),
      num_players_(ParameterValue<int>("players")),
      points_order_(
          ParsePointsOrder(ParameterValue<std::string>("points_order"))),
      returns_type_(
          ParseReturnsType(ParameterValue<std::string>("returns_type"))),
      impinfo_(ParameterValue<bool>("imp_info")) {
  // Override the zero-sum utility in the game type if general-sum returns.
  if (returns_type_ == ReturnsType::kTotalPoints) {
    game_type_.utility = GameType::Utility::kGeneralSum;
  }
  // Maybe override the perfect information in the game type.
  if (impinfo_) {
    game_type_.information = GameType::Information::kImperfectInformation;
  }

  default_observer_ = std::make_shared<GoofspielObserver>(kDefaultObsType);
  info_state_observer_ = std::make_shared<GoofspielObserver>(kInfoStateObsType);
  private_observer_ = std::make_shared<GoofspielObserver>(
      IIGObservationType{.public_info = false,
          .perfect_recall = false,
          .private_info = PrivateInfoType::kSinglePlayer});
  public_observer_ = std::make_shared<GoofspielObserver>(
      IIGObservationType{.public_info = true,
          .perfect_recall = false,
          .private_info = PrivateInfoType::kNone});

}

std::unique_ptr<State> GoofspielGame::NewInitialState() const {
  return std::unique_ptr<State>(new GoofspielState(
      shared_from_this(), num_cards_, points_order_, impinfo_, returns_type_));
}

int GoofspielGame::MaxChanceOutcomes() const {
  if (points_order_ == PointsOrder::kRandom) {
    return num_cards_;
  } else {
    return 0;
  }
}

std::vector<int> GoofspielGame::InformationStateTensorShape() const {
  return InferTensorShape(*this, info_state_observer_);
}

std::vector<int> GoofspielGame::ObservationTensorShape() const {
  return InferTensorShape(*this, default_observer_);
}

double GoofspielGame::MinUtility() const {
  if (returns_type_ == ReturnsType::kWinLoss) {
    return -1;
  } else if (returns_type_ == ReturnsType::kPointDifference) {
    // 0 - (1 + 2 + ... + K) / n
    return -(num_cards_ * (num_cards_ + 1) / 2) / num_players_;
  } else if (returns_type_ == ReturnsType::kTotalPoints) {
    return 0;
  } else {
    SpielFatalError("Unrecognized returns type.");
  }
}

double GoofspielGame::MaxUtility() const {
  if (returns_type_ == ReturnsType::kWinLoss) {
    return 1;
  } else if (returns_type_ == ReturnsType::kPointDifference) {
    // (1 + 2 + ... + K) - (1 + 2 + ... + K) / n
    // = (n-1) (1 + 2 + ... + K) / n
    double sum = num_cards_ * (num_cards_ + 1) / 2;
    return (num_players_ - 1) * sum / num_players_;
  } else if (returns_type_ == ReturnsType::kTotalPoints) {
    // 1 + 2 + ... + K.
    return num_cards_ * (num_cards_ + 1) / 2;
  } else {
    SpielFatalError("Unrecognized returns type.");
  }
}
std::shared_ptr<Observer>
GoofspielGame::MakeObserver(absl::optional<IIGObservationType> iig_obs_type,
                            const GameParameters& params) const {
  if (!params.empty()) SpielFatalError("Observation params not supported");
  return std::make_shared<GoofspielObserver>(iig_obs_type
                                                 .value_or(kDefaultObsType));
}

}  // namespace goofspiel
}  // namespace open_spiel
