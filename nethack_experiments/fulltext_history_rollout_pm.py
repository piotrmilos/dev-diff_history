import argparse
import json
import os
import torch
import functools

from nle.env import tasks
from transformers import GenerationConfig
from copy import deepcopy


from utils import (
    load_hf_lm_and_tokenizer,
    pretty_print_ttyrec,
    set_seed_everywhere,
    UnrollLengthCriteria,
)
from wrappers import NLELMWrapper

special_tokens_interaction_history = {
    "action": "<|action|>",
    "observation": "<|observation|>",
}

ACTION_TOKEN = special_tokens_interaction_history["action"]
OBSERVATION_TOKEN = special_tokens_interaction_history["observation"]


class TrajectoryTokenizer:
    def __init__(self, tokenizer, nethack_anchor_every=1,
                 nethack_obs_start_token_id=30001,
                 nethack_obs_end_token_id=30002,
                 nethack_obs_start_diff_token_id=30004, ):
        self.nethack_obs_start_token_id = nethack_obs_start_token_id
        self.nethack_obs_end_token_id = nethack_obs_end_token_id
        self.nethack_obs_start_diff_token_id = nethack_obs_start_diff_token_id
        self.nethack_anchor_every = nethack_anchor_every

        self.tokenizer = tokenizer

        self._observations = None
        self._actions = None
        self._token_buffer = None
        self._anchor_obs = None

        self.reset()

    def append_observation(self, observation):
        obs = observation.strip()
        print("\033[92m <>" + obs + "</>\033[0m")
        i = len(self._observations)
        self._observations.append(obs)
        assert len(self._observations) == len(self._actions) + 1

        obs = obs.strip()
        if i % self.nethack_anchor_every == 0:
            # Anchor the observation (encode full observation)
            self._anchor_obs = obs
            tokens_obs = self.tokenizer.encode(obs, add_special_tokens=False)
            tokens_obs = [self.nethack_obs_start_token_id] + tokens_obs
        else:
            diff_obs = self.diff_strings(self._anchor_obs, obs) # todo: implement diff_strings
            tokens_obs = self.tokenizer.encode(diff_obs)
            tokens_obs = [self.nethack_obs_start_diff_token_id] + tokens_obs

        tokens_obs += [self.nethack_obs_end_token_id]

        self._token_buffer.extend(tokens_obs)

    def append_action(self, action):
        # print in red
        action = action.strip()
        print("\033[91m <>" + action + "</>\033[0m")
        action = action.strip()
        self._actions.append(action)
        assert len(self._observations) == len(self._actions)
        tokens_action = self.tokenizer.encode(action, add_special_tokens=False)

        self._token_buffer.extend(tokens_action)

    def reset(self):
        self._observations = []
        self._actions = []
        self._token_buffer = [self.tokenizer.bos_token_id]
        self._anchor_obs = None

    def return_tokenized(self):
        assert self._token_buffer[-1] == self.nethack_obs_end_token_id
        return self._token_buffer

def history_rollout(
    model,
    tokenizer,
    action_generation_config,
    args,
    observation_token=OBSERVATION_TOKEN,
    max_tries=1,
    history=2,
    max_ctx_tokens=4000,
    history_max=False,
    start_seed=10000,
):
    game_seed = len([fn for fn in os.listdir(args.ttyrec_save_dir)]) + start_seed

    # instantiate env
    env = tasks.NetHackChallenge(
        **dict(
            savedir=os.path.join(args.ttyrec_save_dir, f"{game_seed}"),
            character="@",
            max_episode_steps=100000000,
            observation_keys=(
                "blstats",
                "tty_chars",
                "tty_cursor",
                "glyphs",
                "inv_strs",
                "inv_letters",
            ),
            penalty_step=0.0,
            penalty_time=0.0,
            penalty_mode="constant",
            no_progress_timeout=100,
            save_ttyrec_every=1,
        )
    )


    env = NLELMWrapper(
        env, observation=True, random_template=False, include_interleave_in_prompt=False
    )
    # env.env.seed(game_seed, game_seed) # pm change_here

    interleaving_token = env.interleaving_token
    observation_token_id = tokenizer.encode_plus(observation_token)["input_ids"][-1]


    def _query_model(
        prompt, unroll_length=1, stop_token_id=observation_token_id
    ):
        stopping_criteria = UnrollLengthCriteria(
            unroll_length=unroll_length,
            stop_token_id=stop_token_id,
            num_return_sequences=1,
        )

        # print in blue
        print("\033[94m <>" + prompt + "</>\033[0m")
        tokenized_prompt = tokenizer(
            prompt, padding="longest", return_tensors="pt", add_special_tokens=False
        )

        if torch.cuda.is_available():
            tokenized_prompt = {k: v.cuda() for (k, v) in tokenized_prompt.items()}

        tries = 0
        while 1:
            tries += 1
            out = model.generate(
                **tokenized_prompt,
                generation_config=action_generation_config,
                stopping_criteria=[stopping_criteria],
            )
            decoded = tokenizer.batch_decode(out)
            suffix = decoded[0][prompt.rfind(ACTION_TOKEN) :]
            if suffix.count(ACTION_TOKEN) > 0:
                break
            if tries > max_tries:
                return ["esc"], ["none"], decoded[0]

        actions = []
        pred_obs = []
        while len(actions) < unroll_length:
            saction = suffix[suffix.find(ACTION_TOKEN) + len(ACTION_TOKEN) :]
            action = (
                saction[: saction.find(OBSERVATION_TOKEN)]
                .replace(ACTION_TOKEN, "")
                .strip()
            )
            if "<" in action:
                action = action[: action.find("<")]
            actions += [action]

            suffix = suffix[suffix.find(OBSERVATION_TOKEN) :]
            obs = suffix[len(OBSERVATION_TOKEN) : suffix.find(ACTION_TOKEN)]
            pred_obs += [obs]
            suffix = suffix[suffix.find(ACTION_TOKEN) :]

        return [actions[-1]], [pred_obs[-1]], decoded[0]

    done = False

    obs = env.reset()

    trajectory_tokenizer = TrajectoryTokenizer(tokenizer, nethack_anchor_every=1)

    query = obs["prompt"]
    ctx = query[:].strip() + "\n"
    ctx_idx = 0
    prompt_history = [obs["prompt"]]

    trajectory_tokenizer.append_observation(ctx)

    imagined_obs = []
    gt_obs = []
    all_obs = []
    all_actions = []
    while not done:
        if history_max:
            passed = False
            candidate_history = min(history, len(all_actions))

            while not passed and candidate_history >= 0:
                query = prompt_history[-(candidate_history + 1)].strip()
                ctx = query.strip() + "\n"
                for i in range(-candidate_history, 0):
                    ctx += "\n%s%s" % (interleaving_token, all_actions[i])
                    ctx += "\n%s\n%s" % (
                        observation_token,
                        prompt_history[i][
                            prompt_history[i].find("statistics[") :
                        ].rstrip("\n"),
                    )
                tokenized_prompt = tokenizer(
                    ctx + "\n%s" % (interleaving_token),
                    padding="longest",
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                if tokenized_prompt["input_ids"].shape[1] < max_ctx_tokens:
                    passed = True
                else:
                    candidate_history -= 1

        else:
            candidate_history = min(history, len(all_actions))
            query = prompt_history[-(candidate_history + 1)].strip()
            ctx = query.strip() + "\n"

            for i in range(-candidate_history, 0):
                ctx += "\n%s%s" % (interleaving_token, all_actions[i])
                ctx += "\n%s\n%s" % (
                    observation_token,
                    prompt_history[i][prompt_history[i].find("statistics[") :].rstrip(
                        "\n"
                    ),
                )

        ctx += "\n%s" % (interleaving_token)

        query_tokens = trajectory_tokenizer.return_tokenized()
        actions, i_obs, decoded_output = _query_model(ctx, unroll_length=1)

        imagined_obs += [i_obs]

        for action in actions:
            try:
                obs, reward, done, info = env.step(action)
                trajectory_tokenizer.append_action(action)
                trajectory_tokenizer.append_observation(obs["prompt"])

                gt_obs += [deepcopy(obs["prompt"])]
                all_actions += [action]
                prompt_history += [obs["prompt"]]

                sobs = pretty_print_ttyrec(obs)
                all_obs += [sobs]
            except Exception as e:
                print("Exception here 1:", e)
                done = True
                try:
                    env.env._quit_game(env.last_observation, done)
                    env.env._quit_game(env.last_observation, done)
                except Exception as e:
                    print("Exception here 2:", e)

        if max(len(all_actions) - history, 0) != 0:
            ctx_idx += 1

        query = prompt_history[ctx_idx].strip()

    obs_mtdata = {
        "gt": gt_obs,
        "pred": imagined_obs,
        "obs": all_obs,
        "actions": all_actions,
    }

    with open(os.path.join(args.observation_save_dir, f"{game_seed}.json"), "w") as f:
        json.dump(json.dumps(obs_mtdata), f)

    try:
        env.env._quit_game(env.last_observation, done)
        env.env._quit_game(env.last_observation, done)
    except:
        return


def main():
    parser = argparse.ArgumentParser()

    # main args
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="experiment_outputs/",
    )
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pred_horizon", type=int, default=8)
    parser.add_argument(
        "--base_dir",
        type=str,
        default="eval_outputs/",
    )
    parser.add_argument("--n_rollouts", type=int, default=256)
    parser.add_argument("--beam_decoding", type=bool, default=True)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--greedy_decoding", type=bool, default=False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--auto_menu_closing", action="store_true")
    parser.add_argument("--history", default=15)
    parser.add_argument("--history_max", action="store_true")

    # fill in later
    parser.add_argument("--ttyrec_save_dir", default=None)
    parser.add_argument("--observation_save_dir", default=None)

    # parse args
    args = parser.parse_args()

    assert not (args.beam_decoding and args.greedy_decoding)

    try:
        args.history = int(args.history)
    except:
        pass

    # fill in fillable args
    if not args.debug:
        model_rollout_dump_dir = os.path.join(
            args.base_dir,
            "-".join(
                args.model_name_or_path.split("/")[-2:]
                + ["ph%i" % args.pred_horizon]
                + ["beam"] * int(args.beam_decoding)
                + [f"nbeams{args.num_beams}"] * int(args.beam_decoding)
                + ["greedy"] * int(args.greedy_decoding)
                + ["autoclose"] * int(args.auto_menu_closing)
            ),
        )

        if args.history > 0:
            model_rollout_dump_dir += f"-hist{args.history}"
        if args.history_max:
            model_rollout_dump_dir += "-histmax"

        if not os.path.exists(model_rollout_dump_dir):
            os.makedirs(model_rollout_dump_dir, exist_ok=True)

        observation_dump_dir = model_rollout_dump_dir + "-observation"

        if not os.path.exists(observation_dump_dir):
            os.makedirs(observation_dump_dir)

        args.ttyrec_save_dir = model_rollout_dump_dir
        args.observation_save_dir = observation_dump_dir
    else:
        args.ttyrec_save_dir = os.path.join(args.base_dir, "dummy_ttyrec")
        args.observation_save_dir = os.path.join(args.base_dir, "dummy_observation")
        if not os.path.exists(args.ttyrec_save_dir):
            os.makedirs(args.ttyrec_save_dir)
        if not os.path.exists(args.observation_save_dir):
            os.makedirs(args.observation_save_dir)

    # set seed everywhere
    set_seed_everywhere(args.seed)

    # is on MacOS, stay on cpu (some type compatibility issues)
    if torch.backends.mps.is_available():
        device = 'cpu'
    else:
        device = 'auto'

    model, tokenizer = load_hf_lm_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        use_fast_tokenizer=(
            "bert" in args.model_name_or_path or "LED" in args.model_name_or_path
        ),
        device_map=device
    )

    if args.beam_decoding:
        action_generation_config = GenerationConfig(
            max_new_tokens=4096,
            decoder_start_token_id=0,
            eos_token_id=model.config.eos_token_id,
            pad_token=model.config.pad_token_id,
            num_beams=args.num_beams,
        )
    else:
        action_generation_config = GenerationConfig(
            max_new_tokens=4096,
            decoder_start_token_id=0,
            eos_token_id=model.config.eos_token_id,
            pad_token=model.config.pad_token_id,
        )

    rollout_fn = functools.partial(
        history_rollout,
        history=args.history,
        model=model,
        tokenizer=tokenizer,
        action_generation_config=action_generation_config,
        args=args,
        history_max=args.history_max,
    )

    for _ in range(args.n_rollouts):
        rollout_fn()


if __name__ == "__main__":
    main()
