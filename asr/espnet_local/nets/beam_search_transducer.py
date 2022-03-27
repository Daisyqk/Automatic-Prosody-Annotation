"""Search algorithms for transducer models."""

from typing import List
from typing import Union

import numpy as np
import torch

from espnet.nets.pytorch_backend.transducer.utils import create_lm_batch_state
from espnet.nets.pytorch_backend.transducer.utils import init_lm_state
from espnet.nets.pytorch_backend.transducer.utils import is_prefix
from espnet.nets.pytorch_backend.transducer.utils import recombine_hyps
from espnet.nets.pytorch_backend.transducer.utils import select_lm_state
from espnet.nets.pytorch_backend.transducer.utils import substract
from espnet.nets.transducer_decoder_interface import Hypothesis
from espnet.nets.transducer_decoder_interface import NSCHypothesis
from espnet.nets.transducer_decoder_interface import TransducerDecoderInterface
from espnet.nets.scorers.mmi_rescorer import MMIRescorer
from espnet.nets.scorers.mmi_rnnt_scorer import MMIRNNTScorer
from espnet.nets.scorers.ctc_rnnt_scorer import CTCRNNTScorer
from espnet.nets.scorers.mmi_rnnt_lookahead_scorer import MMIRNNTLookaheadScorer

class BeamSearchTransducer:
    """Beam search implementation for transducer."""

    def __init__(
        self,
        decoder: Union[TransducerDecoderInterface, torch.nn.Module],
        joint_network: torch.nn.Module,
        beam_size: int,
        lm: torch.nn.Module = None,
        lm_weight: float = 0.1,
        search_type: str = "default",
        char_list = None,
        max_sym_exp: int = 2,
        u_max: int = 50,
        nstep: int = 1,
        prefix_alpha: int = 1,
        score_norm: bool = True,
        nbest: int = 1,
        mmi_scorer=None,
        mmi_weight=0.0,
        ctc_module=None,
        ctc_weight=0.0,
        ngram_scorer=None,
        ngram_weight=0.0,
        word_ngram_scorer=None,
        word_ngram_weight=0.0,
        tlg_scorer=None,
        tlg_weight=0.0,
    ):
        """Initialize transducer beam search.

        Args:
            decoder: Decoder class to use
            joint_network: Joint Network class
            beam_size: Number of hypotheses kept during search
            lm: LM class to use
            lm_weight: lm weight for soft fusion
            search_type: type of algorithm to use for search
            max_sym_exp: number of maximum symbol expansions at each time step ("tsd")
            u_max: maximum output sequence length ("alsd")
            nstep: number of maximum expansion steps at each time step ("nsc")
            prefix_alpha: maximum prefix length in prefix search ("nsc")
            score_norm: normalize final scores by length ("default")
            nbest: number of returned final hypothesis
        """
        self.decoder = decoder
        self.joint_network = joint_network

        self.beam_size = beam_size
        self.hidden_size = decoder.dunits
        self.vocab_size = decoder.odim
        self.blank = decoder.blank

        # MMI alignment scorer
        self.mmi_scorer = mmi_scorer
        self.mmi_weight = mmi_weight

        # deprecated. CTC scorer
        self.ctc_module = ctc_module
        self.ctc_weight = ctc_weight

        # character-level Ngram scorer implemented by kenlm
        self.ngram_scorer = ngram_scorer
        self.ngram_weight = ngram_weight

        # word-level Ngram scorer implemented by k2
        self.word_ngram_scorer = word_ngram_scorer
        self.word_ngram_weight = word_ngram_weight

        # word-level Ngram scorer implemented by pykaldi (cweng)
        self.tlg_scorer = tlg_scorer
        self.tlg_weight = tlg_weight

        if self.beam_size <= 1:
            self.search_algorithm = self.greedy_search
        elif search_type == "default":
            self.search_algorithm = self.default_beam_search
        elif search_type == "tsd":
            self.search_algorithm = self.time_sync_decoding
        elif search_type == "alsd":
            self.search_algorithm = self.align_length_sync_decoding
        elif search_type == "nsc":
            self.search_algorithm = self.nsc_beam_search
        else:
            raise NotImplementedError

        self.lm = lm
        self.lm_weight = lm_weight
        print(f"Using LM {lm} with weight {lm_weight}")

        if lm is not None:
            self.use_lm = True
            self.is_wordlm = True if hasattr(lm.predictor, "wordlm") else False
            self.lm_predictor = lm.predictor.wordlm if self.is_wordlm else lm.predictor
            self.lm_layers = len(self.lm_predictor.rnn)
        else:
            self.use_lm = False

        self.max_sym_exp = max_sym_exp
        self.u_max = u_max
        self.nstep = nstep
        self.prefix_alpha = prefix_alpha
        self.score_norm = score_norm

        self.nbest = nbest
        self.char_list = char_list

    def __call__(self, h: torch.Tensor) -> Union[List[Hypothesis], List[NSCHypothesis]]:
        """Perform beam search.

        Args:
            h: Encoded speech features (T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results

        """
        self.decoder.set_device(h.device)

        if not hasattr(self.decoder, "decoders"):
            self.decoder.set_data_type(h.dtype)

        nbest_hyps = self.search_algorithm(h)
        
        if isinstance(self.mmi_scorer, MMIRescorer):
            nbest_hyps = self.mmi_scorer.score(h, nbest_hyps, v2=True)
        return nbest_hyps

    def sort_nbest(
        self, hyps: Union[List[Hypothesis], List[NSCHypothesis]]
    ) -> Union[List[Hypothesis], List[NSCHypothesis]]:
        """Sort hypotheses by score or score given sequence length.

        Args:
            hyps: list of hypotheses

        Return:
            hyps: sorted list of hypotheses

        """
        if self.score_norm:
            hyps.sort(key=lambda x: x.score / len(x.yseq), reverse=True)
        else:
            hyps.sort(key=lambda x: x.score, reverse=True)

        return hyps[: self.nbest]

    def greedy_search(self, h: torch.Tensor) -> List[Hypothesis]:
        """Greedy search implementation for transformer-transducer.

        Args:
            h: Encoded speech features (T_max, D_enc)

        Returns:
            hyp: 1-best decoding results

        """
        dec_state = self.decoder.init_state(1)

        hyp = Hypothesis(score=0.0, yseq=[self.blank], dec_state=dec_state)
        cache = {}

        y, state, _ = self.decoder.score(hyp, cache)

        for i, hi in enumerate(h):
            ytu = torch.log_softmax(self.joint_network(hi, y), dim=-1)
            logp, pred = torch.max(ytu, dim=-1)
            if pred != self.blank:
                hyp.yseq.append(int(pred))
                hyp.score += float(logp)

                hyp.dec_state = state

                y, state, _ = self.decoder.score(hyp, cache)
        return [hyp]

    def default_beam_search(self, h: torch.Tensor) -> List[Hypothesis]:
        """Beam search implementation.

        Args:
            x: Encoded speech features (T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results

        """
        beam = min(self.beam_size, self.vocab_size)
        beam_k = min(beam, (self.vocab_size - 1))

        dec_state = self.decoder.init_state(1)

        kept_hyps = [Hypothesis(score=0.0, yseq=[self.blank], dec_state=dec_state)]
        cache = {}

        for hi in h:
            hyps = kept_hyps
            kept_hyps = []

            while True:
                max_hyp = max(hyps, key=lambda x: x.score)
                hyps.remove(max_hyp)

                y, state, lm_tokens = self.decoder.score(max_hyp, cache)

                ytu = torch.log_softmax(self.joint_network(hi, y), dim=-1)
                top_k = ytu[1:].topk(beam_k, dim=-1)
                
                # add a blank only
                kept_hyps.append(
                    Hypothesis(
                        score=(max_hyp.score + float(ytu[0:1])),
                        yseq=max_hyp.yseq[:],
                        dec_state=max_hyp.dec_state,
                        lm_state=max_hyp.lm_state,
                    )
                )

                if self.use_lm:
                    lm_state, lm_scores = self.lm.predict(max_hyp.lm_state, lm_tokens)
                else:
                    lm_state = max_hyp.lm_state

                for logp, k in zip(*top_k):
                    score = max_hyp.score + float(logp)

                    if self.use_lm:
                        score += self.lm_weight * lm_scores[0][k + 1]

                    hyps.append(
                        Hypothesis(
                            score=score,
                            yseq=max_hyp.yseq[:] + [int(k + 1)],
                            dec_state=state,
                            lm_state=lm_state,
                        )
                    )

                hyps_max = float(max(hyps, key=lambda x: x.score).score)
                kept_most_prob = sorted(
                    [hyp for hyp in kept_hyps if hyp.score > hyps_max],
                    key=lambda x: x.score,
                )
                if len(kept_most_prob) >= beam:
                    kept_hyps = kept_most_prob
                    break

        return self.sort_nbest(kept_hyps)

    def time_sync_decoding(self, h: torch.Tensor) -> List[Hypothesis]:
        """Time synchronous beam search implementation.

        Based on https://ieeexplore.ieee.org/document/9053040

        Args:
            h: Encoded speech features (T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results

        """
        beam = min(self.beam_size, self.vocab_size)

        beam_state = self.decoder.init_state(beam)

        B = [
            Hypothesis(
                yseq=[self.blank],
                score=0.0,
                dec_state=self.decoder.select_state(beam_state, 0),
            )
        ]
        cache = {}

        if self.use_lm and not self.is_wordlm:
            B[0].lm_state = init_lm_state(self.lm_predictor)

        for hi in h:
            A = []
            C = B

            h_enc = hi.unsqueeze(0)

            for v in range(self.max_sym_exp):
                D = []

                beam_y, beam_state, beam_lm_tokens = self.decoder.batch_score(
                    C,
                    beam_state,
                    cache,
                    self.use_lm,
                )

                beam_logp = torch.log_softmax(self.joint_network(h_enc, beam_y), dim=-1)
                beam_topk = beam_logp[:, 1:].topk(beam, dim=-1)

                seq_A = [h.yseq for h in A]

                for i, hyp in enumerate(C):
                    if hyp.yseq not in seq_A:
                        A.append(
                            Hypothesis(
                                score=(hyp.score + float(beam_logp[i, 0])),
                                yseq=hyp.yseq[:],
                                dec_state=hyp.dec_state,
                                lm_state=hyp.lm_state,
                            )
                        )
                    else:
                        dict_pos = seq_A.index(hyp.yseq)

                        A[dict_pos].score = np.logaddexp(
                            A[dict_pos].score, (hyp.score + float(beam_logp[i, 0]))
                        )

                if v < (self.max_sym_exp - 1):
                    if self.use_lm:
                        beam_lm_states = create_lm_batch_state(
                            [c.lm_state for c in C], self.lm_layers, self.is_wordlm
                        )

                        beam_lm_states, beam_lm_scores = self.lm.buff_predict(
                            beam_lm_states, beam_lm_tokens, len(C)
                        )

                    for i, hyp in enumerate(C):
                        for logp, k in zip(beam_topk[0][i], beam_topk[1][i] + 1):
                            new_hyp = Hypothesis(
                                score=(hyp.score + float(logp)),
                                yseq=(hyp.yseq + [int(k)]),
                                dec_state=self.decoder.select_state(beam_state, i),
                                lm_state=hyp.lm_state,
                            )

                            if self.use_lm:
                                new_hyp.score += self.lm_weight * beam_lm_scores[i, k]

                                new_hyp.lm_state = select_lm_state(
                                    beam_lm_states, i, self.lm_layers, self.is_wordlm
                                )

                            D.append(new_hyp)

                C = sorted(D, key=lambda x: x.score, reverse=True)[:beam]

            B = sorted(A, key=lambda x: x.score, reverse=True)[:beam]

        return self.sort_nbest(B)

    def align_length_sync_decoding(self, h: torch.Tensor) -> List[Hypothesis]:
        """Alignment-length synchronous beam search implementation.

        Based on https://ieeexplore.ieee.org/document/9053040

        Args:
            h: Encoded speech features (T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results

        """
        beam = min(self.beam_size, self.vocab_size)

        h_length = int(h.size(0))
        u_max = min(self.u_max, (h_length - 1))

        beam_state = self.decoder.init_state(beam)        
 
        B = [
            Hypothesis(
                yseq=[self.blank],
                score=0.0,
                dec_state=self.decoder.select_state(beam_state, 0),
                mmi_tot_score=0.0,
                word_ngram_score=0.0,
                tlg_state=self.tlg_scorer.init_state() if self.tlg_scorer else None
            )
        ]
        # final hypothesis set to return
        final = []
        # For hypothesis with same yseq, its decoder output could be cached
        # yseq -> decoder_out, decoder_state
        cache = {} 

        # lm initialization
        if self.use_lm and not self.is_wordlm:
            B[0].lm_state = init_lm_state(self.lm_predictor)
 
        if isinstance(self.mmi_scorer, (MMIRNNTScorer, CTCRNNTScorer, MMIRNNTLookaheadScorer)):
            mmi_nnet_output, mmi_den_scores = self.mmi_scorer.den_scores(h)

        if self.ctc_module:
            ctc_pred = self.ctc_module.log_softmax(h.unsqueeze(0))[0]

        for tu_sum in range(h_length + u_max):
            A = [] # collection for next step
            B_ = [] # collection for search in this step. state of pred. net is kept
            h_states = [] # collection of encoder_out frame for each hypothesis
            p_ctc = [] # collection of ctc distribution
            for j, hyp in enumerate(B): # skip all hypothesis that head the last frame.
                u = len(hyp.yseq) - 1
                t = tu_sum - u + 1

                if t > (h_length - 1):
                    continue

                B_.append(hyp)
                h_states.append((t, h[t]))

                if self.ctc_module:
                    p_ctc.append(ctc_pred[t])

            if B_:
                beam_y, beam_state, beam_lm_tokens = self.decoder.batch_score(
                    B_,
                    beam_state,
                    cache,
                    self.use_lm,
                )
                
                h_enc = torch.stack([h[1] for h in h_states])

                # [beam, h_dim], [beam, h_dim]
                beam_logp = torch.log_softmax(self.joint_network(h_enc, beam_y), dim=-1) # [beam, vocab]
               
                if self.ctc_module:
                    p_ctc = torch.stack([p for p in p_ctc])
                    beam_logp += self.ctc_weight * p_ctc

                # warning: like in LASCTC, the LM score would not be considered in top-k process
                beam_topk = beam_logp[:, 1:].topk(beam, dim=-1) # values and indices: [beam, beam]. blank excluded

                if self.use_lm:
                    beam_lm_states = create_lm_batch_state(
                        [b.lm_state for b in B_], self.lm_layers, self.is_wordlm
                    )

                    # LM prediction for next token
                    beam_lm_states, beam_lm_scores = self.lm.buff_predict(
                        beam_lm_states, beam_lm_tokens, len(B_)
                    )

                for i, hyp in enumerate(B_):
                    # tot_mmi_score in both case just keep the history
                    # but will be re-assigned later

                    # keep all hyps with an additional blank
                    new_hyp = Hypothesis(
                        score=(hyp.score + float(beam_logp[i, 0])),
                        yseq=hyp.yseq[:],
                        dec_state=hyp.dec_state,
                        lm_state=hyp.lm_state,
                        mmi_tot_score=hyp.mmi_tot_score,
                        word_ngram_score=hyp.word_ngram_score,
                        tlg_state=hyp.tlg_state,
                    )

                    A.append(new_hyp)

                    if h_states[i][0] == (h_length - 1): # heat the last frame
                        final.append(new_hyp)

                    # Only search a part of candidate tokens
                    if self.word_ngram_scorer and self.word_ngram_weight > 0.0:
                        next_tokens = beam_topk[1][i] + 1
                        word_ngram_scores, word_ngram_states = self.word_ngram_scorer.score_partial(
                                                                 hyp.yseq[1:], next_tokens, 
                                                                 hyp.word_ngram_score, None)
                    else:
                        word_ngram_scores = [0.0] * len(beam_topk[1][i])
                        word_ngram_states = [0.0] * len(beam_topk[1][i])

                    if self.tlg_scorer and self.tlg_weight > 0.0:
                        next_tokens = beam_topk[1][i] + 1
                        tlg_scores, tlg_states = self.tlg_scorer.score_partial(
                                                     None, next_tokens,
                                                     hyp.tlg_state, None)
                    else:
                        tlg_scores = [0.0] * len(beam_topk[1][i])
                        tlg_states = [None] * len(beam_topk[1][i])
                     
                    for j, (logp, k) in enumerate(zip(beam_topk[0][i], beam_topk[1][i] + 1)):
 
                        new_hyp = Hypothesis(
                            score=(hyp.score + float(logp)),
                            yseq=(hyp.yseq[:] + [int(k)]),
                            dec_state=self.decoder.select_state(beam_state, i),
                            lm_state=hyp.lm_state,
                            mmi_tot_score=hyp.mmi_tot_score,
                            word_ngram_score=word_ngram_states[j],
                            tlg_state=tlg_states[j] 
                        )

                        # add LM scores. possibly 4 styles
                        if self.use_lm:
                            new_hyp.score += self.lm_weight * beam_lm_scores[i, k]

                            new_hyp.lm_state = select_lm_state(
                                beam_lm_states, i, self.lm_layers, self.is_wordlm
                            )

                        if self.word_ngram_scorer and self.word_ngram_weight > 0.0:
                            new_hyp.score += self.word_ngram_weight * word_ngram_scores[j]

                        if self.tlg_scorer and self.tlg_weight > 0.0:
                            new_hyp.score += self.tlg_weight * tlg_scores[j]

                        if self.ngram_scorer and self.ngram_weight > 0.0:
                            ngram_score, _ = self.ngram_scorer.score_partial(
                                             torch.Tensor(hyp.yseq[:]).int(),
                                             torch.Tensor([int(k)]).int(), 
                                             None, h)
                            new_hyp.score += self.ngram_weight * ngram_score.item()
                            #text = "".join([self.mmi_scorer.char_list[x] for x in new_hyp.yseq[1:]])
                            #print(f"text: {text} | ngram_score: {ngram_score.item()}", flush=True)
                            
                        A.append(new_hyp)
            
            if isinstance(self.mmi_scorer, (MMIRNNTScorer, CTCRNNTScorer, MMIRNNTLookaheadScorer)) and self.mmi_weight > 0.0:
                # any hypo in A has t + u = i + 1: a new step
                A = self.mmi_scorer.batch_score(A, mmi_nnet_output, mmi_den_scores, tu_sum+1, self.mmi_weight)

            # unlike the original implementation, we combine the hypotheses before pruning
            # this allow the hypothesis different and possibly make the rescore more effective
            B = recombine_hyps(A, self.mmi_weight)
            B = sorted(B, key=lambda x: x.score, reverse=True)[:beam]
            
            #print("#"*20, f"tu_sum: {tu_sum}","#"*20)
            #for hy in B:
            #    if hasattr(self.mmi_scorer, "char_list"):
                    # text = "".join([self.mmi_scorer.char_list[x] for x in hy.yseq[1:]])
                    # print(f"text: {text} | score: {hy.score} | mmi_score: {hy.mmi_tot_score} | rnnt_score: {hy.score - hy.mmi_tot_score * self.mmi_weight}", flush=True)
        if self.tlg_scorer and self.tlg_weight > 0.0 and final:
            tlg_final_states = [h.tlg_state for h in final]
            tlg_final_scores = self.tlg_scorer.final_score(tlg_final_states)
            for i, h in enumerate(final):
                h.score += self.tlg_weight * tlg_final_scores[i]

        if final:
            return self.sort_nbest(final)
        else:
            print("Warning: No finished hypothesis found. return the partial hypothesis", flush=True)
            return B

    def nsc_beam_search(self, h: torch.Tensor) -> List[NSCHypothesis]:
        """N-step constrained beam search implementation.

        Based and modified from https://arxiv.org/pdf/2002.03577.pdf.
        Please reference ESPnet (b-flo, PR #2444) for any usage outside ESPnet
        until further modifications.

        Note: the algorithm is not in his "complete" form but works almost as
        intended.

        Args:
            h: Encoded speech features (T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results

        """
        beam = min(self.beam_size, self.vocab_size)
        beam_k = min(beam, (self.vocab_size - 1))

        beam_state = self.decoder.init_state(beam)

        init_tokens = [
            NSCHypothesis(
                yseq=[self.blank],
                score=0.0,
                dec_state=self.decoder.select_state(beam_state, 0),
            )
        ]

        cache = {}

        beam_y, beam_state, beam_lm_tokens = self.decoder.batch_score(
            init_tokens,
            beam_state,
            cache,
            self.use_lm,
        )

        state = self.decoder.select_state(beam_state, 0)

        if self.use_lm:
            beam_lm_states, beam_lm_scores = self.lm.buff_predict(
                None, beam_lm_tokens, 1
            )
            lm_state = select_lm_state(
                beam_lm_states, 0, self.lm_layers, self.is_wordlm
            )
            lm_scores = beam_lm_scores[0]
        else:
            lm_state = None
            lm_scores = None

        kept_hyps = [
            NSCHypothesis(
                yseq=[self.blank],
                score=0.0,
                dec_state=state,
                y=[beam_y[0]],
                lm_state=lm_state,
                lm_scores=lm_scores,
            )
        ]

        for hi in h:
            hyps = sorted(kept_hyps, key=lambda x: len(x.yseq), reverse=True)
            kept_hyps = []

            h_enc = hi.unsqueeze(0)

            for j, hyp_j in enumerate(hyps[:-1]):
                for hyp_i in hyps[(j + 1) :]:
                    curr_id = len(hyp_j.yseq)
                    next_id = len(hyp_i.yseq)

                    if (
                        is_prefix(hyp_j.yseq, hyp_i.yseq)
                        and (curr_id - next_id) <= self.prefix_alpha
                    ):
                        ytu = torch.log_softmax(
                            self.joint_network(hi, hyp_i.y[-1]), dim=-1
                        )

                        curr_score = hyp_i.score + float(ytu[hyp_j.yseq[next_id]])

                        for k in range(next_id, (curr_id - 1)):
                            ytu = torch.log_softmax(
                                self.joint_network(hi, hyp_j.y[k]), dim=-1
                            )

                            curr_score += float(ytu[hyp_j.yseq[k + 1]])

                        hyp_j.score = np.logaddexp(hyp_j.score, curr_score)

            S = []
            V = []
            for n in range(self.nstep):
                beam_y = torch.stack([hyp.y[-1] for hyp in hyps])

                beam_logp = torch.log_softmax(self.joint_network(h_enc, beam_y), dim=-1)
                beam_topk = beam_logp[:, 1:].topk(beam_k, dim=-1)

                for i, hyp in enumerate(hyps):
                    S.append(
                        NSCHypothesis(
                            yseq=hyp.yseq[:],
                            score=hyp.score + float(beam_logp[i, 0:1]),
                            y=hyp.y[:],
                            dec_state=hyp.dec_state,
                            lm_state=hyp.lm_state,
                            lm_scores=hyp.lm_scores,
                        )
                    )

                    for logp, k in zip(beam_topk[0][i], beam_topk[1][i] + 1):
                        score = hyp.score + float(logp)

                        if self.use_lm:
                            score += self.lm_weight * float(hyp.lm_scores[k])

                        V.append(
                            NSCHypothesis(
                                yseq=hyp.yseq[:] + [int(k)],
                                score=score,
                                y=hyp.y[:],
                                dec_state=hyp.dec_state,
                                lm_state=hyp.lm_state,
                                lm_scores=hyp.lm_scores,
                            )
                        )

                V.sort(key=lambda x: x.score, reverse=True)
                V = substract(V, hyps)[:beam]

                beam_state = self.decoder.create_batch_states(
                    beam_state,
                    [v.dec_state for v in V],
                    [v.yseq for v in V],
                )
                beam_y, beam_state, beam_lm_tokens = self.decoder.batch_score(
                    V,
                    beam_state,
                    cache,
                    self.use_lm,
                )

                if self.use_lm:
                    beam_lm_states = create_lm_batch_state(
                        [v.lm_state for v in V], self.lm_layers, self.is_wordlm
                    )
                    beam_lm_states, beam_lm_scores = self.lm.buff_predict(
                        beam_lm_states, beam_lm_tokens, len(V)
                    )

                if n < (self.nstep - 1):
                    for i, v in enumerate(V):
                        v.y.append(beam_y[i])

                        v.dec_state = self.decoder.select_state(beam_state, i)

                        if self.use_lm:
                            v.lm_state = select_lm_state(
                                beam_lm_states, i, self.lm_layers, self.is_wordlm
                            )
                            v.lm_scores = beam_lm_scores[i]

                    hyps = V[:]
                else:
                    beam_logp = torch.log_softmax(
                        self.joint_network(h_enc, beam_y), dim=-1
                    )

                    for i, v in enumerate(V):
                        if self.nstep != 1:
                            v.score += float(beam_logp[i, 0])

                        v.y.append(beam_y[i])

                        v.dec_state = self.decoder.select_state(beam_state, i)

                        if self.use_lm:
                            v.lm_state = select_lm_state(
                                beam_lm_states, i, self.lm_layers, self.is_wordlm
                            )
                            v.lm_scores = beam_lm_scores[i]

            kept_hyps = sorted((S + V), key=lambda x: x.score, reverse=True)[:beam]

        return self.sort_nbest(kept_hyps)
