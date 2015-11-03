package nlp.assignments;


import java.util.*;
import java.io.*;

import nlp.io.IOUtils;
import nlp.util.*;
import java.util.HashMap;
import java.util.Map;

/**
 * Harness for testing word-level alignments.  The code is hard-wired for the
 * alignment source to be English and the alignment target to be French (recall
 * that's the direction for translating INTO English in the noisy channel
 * model).
 *
 * Your projects will implement several methods of word-to-word alignment.
 */
public class WordAlignmentTester {

  static final String ENGLISH_EXTENSION = "e";
  static final String FRENCH_EXTENSION = "f";

  /**
   * A holder for a pair of sentences, each a list of strings.  Sentences in
   * the test sets have integer IDs, as well, which are used to retreive the
   * gold standard alignments for those sentences.
   */
  public static class SentencePair {
    int sentenceID;
    String sourceFile;
    List<String> englishWords;
    List<String> frenchWords;

    public int getSentenceID() {
      return sentenceID;
    }

    public String getSourceFile() {
      return sourceFile;
    }

    public List<String> getEnglishWords() {
      return englishWords;
    }

    public List<String> getFrenchWords() {
      return frenchWords;
    }

    public String toString() {
      StringBuilder sb = new StringBuilder();
      for (int englishPosition = 0; englishPosition < englishWords.size(); englishPosition++) {
        String englishWord = englishWords.get(englishPosition);
        sb.append(englishPosition);
        sb.append(":");
        sb.append(englishWord);
        sb.append(" ");
      }
      sb.append("\n");
      for (int frenchPosition = 0; frenchPosition < frenchWords.size(); frenchPosition++) {
        String frenchWord = frenchWords.get(frenchPosition);
        sb.append(frenchPosition);
        sb.append(":");
        sb.append(frenchWord);
        sb.append(" ");
      }
      sb.append("\n");
      return sb.toString();
    }

    public SentencePair(int sentenceID, String sourceFile, List<String> englishWords, List<String> frenchWords) {
      this.sentenceID = sentenceID;
      this.sourceFile = sourceFile;
      this.englishWords = englishWords;
      this.frenchWords = frenchWords;
    }
  }

  /**
   * Alignments serve two purposes, both to indicate your system's guessed
   * alignment, and to hold the gold standard alignments.  Alignments map index
   * pairs to one of three values, unaligned, possibly aligned, and surely
   * aligned.  Your alignment guesses should only contain sure and unaligned
   * pairs, but the gold alignments contain possible pairs as well.
   *
   * To build an alignment, start with an empty one and use
   * addAlignment(i,j,true).  To display one, use the render method.
   */
  public static class Alignment {
    Set<Pair<Integer, Integer>> sureAlignments;
    Set<Pair<Integer, Integer>> possibleAlignments;

    public boolean containsSureAlignment(int englishPosition, int frenchPosition) {
      return sureAlignments.contains(new Pair<Integer, Integer>(englishPosition, frenchPosition));
    }

    public boolean containsPossibleAlignment(int englishPosition, int frenchPosition) {
      return possibleAlignments.contains(new Pair<Integer, Integer>(englishPosition, frenchPosition));
    }

    public void addAlignment(int englishPosition, int frenchPosition, boolean sure) {
      Pair<Integer, Integer> alignment = new Pair<Integer, Integer>(englishPosition, frenchPosition);
      if (sure)
        sureAlignments.add(alignment);
      possibleAlignments.add(alignment);
    }

    public Alignment() {
      sureAlignments = new HashSet<Pair<Integer, Integer>>();
      possibleAlignments = new HashSet<Pair<Integer, Integer>>();
    }

    public static String render(Alignment alignment, SentencePair sentencePair) {
      return render(alignment, alignment, sentencePair);
    }

    public static String render(Alignment reference, Alignment proposed, SentencePair sentencePair) {
      StringBuilder sb = new StringBuilder();
      for (int frenchPosition = 0; frenchPosition < sentencePair.getFrenchWords().size(); frenchPosition++) {
        for (int englishPosition = 0; englishPosition < sentencePair.getEnglishWords().size(); englishPosition++) {
          boolean sure = reference.containsSureAlignment(englishPosition, frenchPosition);
          boolean possible = reference.containsPossibleAlignment(englishPosition, frenchPosition);
          char proposedChar = ' ';
          if (proposed.containsSureAlignment(englishPosition, frenchPosition))
            proposedChar = '#';
          if (sure) {
            sb.append('[');
            sb.append(proposedChar);
            sb.append(']');
          } else {
            if (possible) {
              sb.append('(');
              sb.append(proposedChar);
              sb.append(')');
            } else {
              sb.append(' ');
              sb.append(proposedChar);
              sb.append(' ');
            }
          }
        }
        sb.append("| ");
        sb.append(sentencePair.getFrenchWords().get(frenchPosition));
        sb.append('\n');
      }
      for (int englishPosition = 0; englishPosition < sentencePair.getEnglishWords().size(); englishPosition++) {
        sb.append("---");
      }
      sb.append("'\n");
      boolean printed = true;
      int index = 0;
      while (printed) {
        printed = false;
        StringBuilder lineSB = new StringBuilder();
        for (int englishPosition = 0; englishPosition < sentencePair.getEnglishWords().size(); englishPosition++) {
          String englishWord = sentencePair.getEnglishWords().get(englishPosition);
          if (englishWord.length() > index) {
            printed = true;
            lineSB.append(' ');
            lineSB.append(englishWord.charAt(index));
            lineSB.append(' ');
          } else {
            lineSB.append("   ");
          }
        }
        index += 1;
        if (printed) {
          sb.append(lineSB);
          sb.append('\n');
        }
      }
      return sb.toString();
    }
  }

	/**
	 * WordAligners have one method: alignSentencePair, which takes a sentence
	 * pair and produces an alignment which specifies an english source for each
	 * french word which is not aligned to "null". Explicit alignment to
	 * position -1 is equivalent to alignment to "null".
	 */
	static interface WordAligner {
		Alignment alignSentencePair(SentencePair sentencePair);
        void train(List<SentencePair> trainingSentencePairs);
	}

	/**
	 * Simple alignment baseline which maps french positions to english
	 * positions. If the french sentence is longer, all final word map to null.
	 */

    static class BaselineWordAligner implements WordAligner {
		public Alignment alignSentencePair(SentencePair sentencePair) {
			Alignment alignment = new Alignment();
			int numFrenchWords = sentencePair.getFrenchWords().size();
			int numEnglishWords = sentencePair.getEnglishWords().size();
			for (int frenchPosition = 0; frenchPosition < numFrenchWords; frenchPosition++) {
				int englishPosition = frenchPosition;
				if (englishPosition >= numEnglishWords)
					englishPosition = -1;
				alignment.addAlignment(englishPosition, frenchPosition, true);
			}
			return alignment;
		}

        public void train(List<SentencePair> trainingSentencePair) {}
	}

    static class DiceWordAligner implements WordAligner {
        CounterMap<String, String> PairCount = new CounterMap<String, String>();
        Counter<String> frenchWordCount = new Counter<String>();
        Counter<String> englishWordCount = new Counter<String>();
        
        public void train(List<SentencePair> trainingSentencePairs) {
            
            for (SentencePair sentencePair : trainingSentencePairs) {
                int numFrenchWords = sentencePair.getFrenchWords().size(); 
                int numEnglishWords = sentencePair.getEnglishWords().size();
                for (int frenchPosition = 0; frenchPosition < numFrenchWords; frenchPosition++) {
                    frenchWordCount.incrementCount(sentencePair.getFrenchWords().get(frenchPosition), 1.0);
                    for (int englishPosition = 0; englishPosition < numEnglishWords; englishPosition++) {
                        PairCount.incrementCount(sentencePair.getFrenchWords().get(frenchPosition),
                                sentencePair.getEnglishWords().get(englishPosition), 1.0);
                    }
                }
                for (int englishPosition = 0; englishPosition < numEnglishWords; englishPosition++) {
                    englishWordCount.incrementCount(sentencePair.getEnglishWords().get(englishPosition), 1.0);
                }
            }
        }
        
        public Alignment alignSentencePair(SentencePair sentencePair) {
            

            Alignment alignment = new Alignment();
            int numFrenchWords = sentencePair.getFrenchWords().size();
            int numEnglishWords = sentencePair.getEnglishWords().size();
            
            List<String> frenchWords = sentencePair.getFrenchWords();
            List<String> englishWords = sentencePair.getEnglishWords();
            List<Integer> frenchPositions = new ArrayList<Integer>();
            List<Integer> englishPositions = new ArrayList<Integer>();
            for (int i = 0; i < numFrenchWords; i++) {
                frenchPositions.add(i);
            }
            for (int i = 0; i < numEnglishWords; i++) {
                englishPositions.add(i);
            }
            while (!frenchPositions.isEmpty() && !englishPositions.isEmpty()) {
                double maxScore = Double.NEGATIVE_INFINITY;
                int maxEnglishPosition = -1;
                int maxFrenchPosition = -1;
                int frenchPosition = -1;
                for (String frenchWord : frenchWords) {
                       frenchPosition += 1;
                       if (!frenchPositions.contains(frenchPosition)) {continue;}
                       double frenchCount = frenchWordCount.getCount(frenchWord);
                       int englishPosition = -1;

                       for (String englishWord : englishWords) {
                            englishPosition += 1;
                            if (!englishPositions.contains(englishPosition)) {continue;}
                            double englishCount = englishWordCount.getCount(englishWord);
                            double pair = PairCount.getCounter(frenchWord).getCount(englishWord);
                            double DiceScore = 2*pair/(frenchCount+englishCount);
                            if (DiceScore > maxScore) {
                                maxEnglishPosition = englishPosition;
                                maxFrenchPosition = frenchPosition;
                                maxScore = DiceScore;
                            }
                       }
                }
                if (maxEnglishPosition == -1 && maxFrenchPosition == -1) {break;}
                if (maxEnglishPosition != -1) {
                    englishPositions.remove(englishPositions.indexOf(maxEnglishPosition));
                    frenchPositions.remove(frenchPositions.indexOf(maxFrenchPosition));
                    alignment.addAlignment(maxEnglishPosition, maxFrenchPosition, true);
                }
            }
            for (int frenchPosition : frenchPositions) {
                alignment.addAlignment(-1, frenchPosition, true);
            }
            return alignment;
        }
    }

    static class IBM1WordAligner implements WordAligner {
        CounterMap<String, String> feTCount = new CounterMap<String, String>();
        CounterMap<String, String> efTCount = new CounterMap<String, String>();

        public void train(List<SentencePair> trainingSentencePairs) {
            int iteration = 0;
           
            for (SentencePair sentencePair : trainingSentencePairs) {
                List<String> frenchWords = sentencePair.getFrenchWords();
                List<String> englishWords = sentencePair.getEnglishWords();
                for (String englishWord : englishWords) {
                    int englishNum = englishWords.size();
                    for (String frenchWord : frenchWords) {
                        int frenchNum = frenchWords.size();
                        feTCount.setCount(frenchWord, englishWord, 1.0/(1+englishNum));
                        efTCount.setCount(englishWord, frenchWord, 1.0/(1+frenchNum));
                    }
                }
            }

            while (!(iteration == 5)){
            CounterMap<String, String> feCount = new CounterMap<String, String>();
            Counter<String> fCount = new Counter<String>();

            CounterMap<String, String> efCount = new CounterMap<String, String>();
            Counter<String> eCount = new Counter<String>();
            
            for (SentencePair sentencePair : trainingSentencePairs) {
                List<String> frenchWords = sentencePair.getFrenchWords();
                List<String> englishWords = sentencePair.getEnglishWords();
                
                
                for (String englishWord : englishWords) {
                    double Z = 0.0;
                    for (String frenchWord : frenchWords) {
                        Z += feTCount.getCounter(frenchWord).getCount(englishWord);
                    }
                    for (String frenchWord: frenchWords) {
                        feCount.incrementCount(frenchWord, englishWord, feTCount.getCounter(frenchWord).getCount(englishWord)/Z);
                        fCount.incrementCount(frenchWord, feTCount.getCounter(frenchWord).getCount(englishWord)/Z);
                    }
                }
                
                for (String frenchWord : frenchWords) {
                    double Y = 0.0;
                    for (String englishWord : englishWords) {
                        Y += efTCount.getCounter(englishWord).getCount(frenchWord);
                    }
                    for (String englishWord : englishWords) {
                        efCount.incrementCount(englishWord, frenchWord, efTCount.getCounter(englishWord).getCount(frenchWord)/Y);
                        eCount.incrementCount(englishWord, efTCount.getCounter(englishWord).getCount(frenchWord)/Y);
                    }
                }
            }
            for (String frenchWord : feTCount.keySet()) {
                double c = fCount.getCount(frenchWord);
                for (String englishWord : feTCount.getCounter(frenchWord).keySet()) {
                    double cc = eCount.getCount(englishWord);
                    feTCount.setCount(frenchWord, englishWord, feCount.getCounter(frenchWord).getCount(englishWord)/c);
                    efTCount.setCount(englishWord, frenchWord, efCount.getCounter(englishWord).getCount(frenchWord)/cc);
                }
            }
            iteration += 1;
            }
        }

        public Alignment alignSentencePair(SentencePair sentencePair) {
           /* 
            Alignment alignment = new Alignment();
            List<String> frenchWords = sentencePair.getFrenchWords();
            List<String> englishWords = sentencePair.getEnglishWords();
            for (String frenchWord : frenchWords) {
                double bestp = Double.NEGATIVE_INFINITY;
                int bestj = -1;
                int frenchPosition = frenchWords.indexOf(frenchWord);
                for (String englishWord : englishWords) {
                    if (efTCount.getCounter(englishWord).getCount(frenchWord) > bestp) {
                        bestp = efTCount.getCounter(englishWord).getCount(frenchWord);
                        bestj = englishWords.indexOf(englishWord);
                    }
                }
                alignment.addAlignment(bestj, frenchPosition, true);
            }
            return alignment;
        }
        */
            Alignment alignment = new Alignment();
            int numFrenchWords = sentencePair.getFrenchWords().size();
            int numEnglishWords = sentencePair.getEnglishWords().size();

            List<String> frenchWords = sentencePair.getFrenchWords();
            List<String> englishWords = sentencePair.getEnglishWords();
            List<Integer> frenchPositions = new ArrayList<Integer>();
            List<Integer> englishPositions = new ArrayList<Integer>();
            for (int i = 0; i < numFrenchWords; i++) {
               frenchPositions.add(i); 
            }
            for (int i = 0; i < numEnglishWords; i++) {
                englishPositions.add(i);
            }


            while (!frenchPositions.isEmpty() && !englishPositions.isEmpty()) {
                double maxScore = Double.NEGATIVE_INFINITY;
                int maxEnglishPosition = -1;
                int maxFrenchPosition = -1;
                int frenchPosition = -1;

                for (String frenchWord : frenchWords) {
                    frenchPosition += 1;
                    if (!frenchPositions.contains(frenchPosition)) {continue;}
                    int englishPosition = -1;
                    for (String englishWord : englishWords) {
                        englishPosition += 1;
                        if (!englishPositions.contains(englishPosition)) {continue;}
                        double IBM1Score = (feTCount.getCounter(frenchWord).getCount(englishWord)
                            + efTCount.getCounter(englishWord).getCount(frenchWord));
                        if (IBM1Score > maxScore) {
                            maxEnglishPosition = englishPosition;
                            maxFrenchPosition = frenchPosition;
                            maxScore = IBM1Score;
                        }
                    } 

                }
                if (maxEnglishPosition == -1 && maxFrenchPosition == -1) {break;}
                if (maxEnglishPosition != -1) {
                    englishPositions.remove(englishPositions.indexOf(maxEnglishPosition));
                    frenchPositions.remove(frenchPositions.indexOf(maxFrenchPosition));
                    alignment.addAlignment(maxEnglishPosition, maxFrenchPosition, true);
                }
            }
            for (int frenchPosition : frenchPositions) {
               alignment.addAlignment(-1, frenchPosition, true); 
            }
            return alignment;
        }
    
    }

    static class IBM2WordAligner implements WordAligner {
        
        CounterMap<String, String> feTCount = new CounterMap<String, String>();
        CounterMap<String, String> feQCount = new CounterMap<String, String>();
        CounterMap<String, String> efTCount = new CounterMap<String, String>();
        CounterMap<String, String> efQCount = new CounterMap<String, String>();
        

        public void train(List<SentencePair> trainingSentencePairs) {
            int iteration = 0;
           
            for (SentencePair sentencePair : trainingSentencePairs) {
                List<String> frenchWords = sentencePair.getFrenchWords();
                List<String> englishWords = sentencePair.getEnglishWords();
                int englishPosition = -1;
                int englishNum = englishWords.size();
                String englishLength = Integer.toString(englishNum);
                int frenchNum = frenchWords.size();
                String frenchLength = Integer.toString(frenchNum);
                for (String englishWord : englishWords) {
                    englishPosition += 1;
                    String jPosition = Integer.toString(englishPosition);
                    int frenchPosition = -1;
                    for (String frenchWord : frenchWords) {
                        frenchPosition += 1;
                        String iPosition = Integer.toString(frenchPosition);
                        feTCount.setCount(frenchWord, englishWord, 1.0/(1+frenchNum));
                        efTCount.setCount(englishWord, frenchWord, 1.0/(1+englishNum));
                        feQCount.setCount(frenchLength+'_'+englishLength+'_'+jPosition, iPosition, 1.0/(1+frenchNum));
                        efQCount.setCount(englishLength+'_'+frenchLength+'_'+iPosition, jPosition, 1.0/(1+englishNum));
                    }
                }
            }

            while (!(iteration == 5)){
            Counter<String> feCount = new Counter<String>();
            Counter<String> fCount = new Counter<String>();

            Counter<String> efCount = new Counter<String>();
            Counter<String> eCount = new Counter<String>();
            
            Counter<String> feLength = new Counter<String>();
            Counter<String> fLength = new Counter<String>();

            Counter<String> efLength = new Counter<String>();
            Counter<String> eLength = new Counter<String>();
            
            for (SentencePair sentencePair : trainingSentencePairs) {
                List<String> frenchWords = sentencePair.getFrenchWords();
                List<String> englishWords = sentencePair.getEnglishWords();
                String frenchLength = Integer.toString(frenchWords.size());
                String englishLength = Integer.toString(englishWords.size());
                
                int englishPosition = -1;
                for (String englishWord : englishWords) {
                    double Z = 0.0;
                    englishPosition += 1;
                    String jPosition = Integer.toString(englishPosition);
                    int frenchPosition = -1;
                    for (String frenchWord : frenchWords) {
                        frenchPosition += 1;
                        String iPosition = Integer.toString(frenchPosition);
                        Z += feQCount.getCounter(frenchLength+'_'+englishLength+'_'+jPosition).getCount(iPosition)
                                * feTCount.getCounter(frenchWord).getCount(englishWord);
                    }
                    frenchPosition = -1;
                    for (String frenchWord : frenchWords) {
                        frenchPosition += 1;
                        String iPosition = Integer.toString(frenchPosition);
                        double c = feQCount.getCounter(frenchLength+'_'+englishLength+'_'+jPosition).getCount(iPosition)
                                    * feTCount.getCounter(frenchWord).getCount(englishWord)/Z;
                        feCount.incrementCount(frenchWord+'_'+englishWord, c);
                        fCount.incrementCount(frenchWord, c);
                        feLength.incrementCount(frenchLength+'_'+englishLength+'_'+jPosition+'_'+iPosition, c);
                        fLength.incrementCount(frenchLength+'_'+englishLength+'_'+jPosition, c);
                    }
                }
                
                int frenchPosition = -1;
                for (String frenchWord : frenchWords) {
                    double Y = 0.0;
                    frenchPosition += 1;
                    String iPosition = Integer.toString(frenchPosition);
                    englishPosition = -1;
                    for (String englishWord : englishWords) {
                        englishPosition += 1;
                        String jPosition = Integer.toString(englishPosition);
                        Y += efQCount.getCounter(englishLength+'_'+frenchLength+'_'+iPosition).getCount(jPosition) 
                                * efTCount.getCounter(englishWord).getCount(frenchWord);
                    }
                    englishPosition = -1;
                    for (String englishWord: englishWords) {
                        englishPosition += 1;
                        String jPosition = Integer.toString(englishPosition);
                        double c = efQCount.getCounter(englishLength+'_'+frenchLength+'_'+iPosition).getCount(jPosition)
                                    * efTCount.getCounter(englishWord).getCount(frenchWord)/Y;
                        efCount.incrementCount(englishWord+'_'+frenchWord, c);
                        eCount.incrementCount(englishWord, c);
                        efLength.incrementCount(englishLength+'_'+frenchLength+'_'+iPosition+'_'+jPosition, c);
                        eLength.incrementCount(englishLength+'_'+frenchLength+'_'+iPosition, c);
                    }
                }
                
            }

            for (String englishWord : efTCount.keySet()) {
                double c = eCount.getCount(englishWord);
                for (String frenchWord : efTCount.getCounter(englishWord).keySet()) {
                    double cc = fCount.getCount(frenchWord);
                    efTCount.setCount(englishWord,frenchWord, efCount.getCount(englishWord+'_'+frenchWord)/c);
                    feTCount.setCount(frenchWord,englishWord, feCount.getCount(frenchWord+'_'+englishWord)/cc);
                }
            }
            for (String ilm : efQCount.keySet()) {
                double c = eLength.getCount(ilm);
                for (String jPosition : efQCount.getCounter(ilm).keySet()) {
                    efQCount.setCount(ilm, jPosition, efLength.getCount(ilm+'_'+jPosition)/c);
                }
            }
            /*
            for (String frenchWord : feTCount.keySet()) {
                for (String englishWord : feTCount.getCounter(frenchWord).keySet()) {
                    feTCount.setCount(frenchWord, englishWord, feCount.getCounter(frenchWord).getCount(englishWord)/fCount.getCount(frenchWord));
                }
            }
            */
            for (String jlm : feQCount.keySet()) {
                double c = fLength.getCount(jlm);
                for (String iPosition : feQCount.getCounter(jlm).keySet()) {
                    feQCount.setCount(jlm, iPosition, feLength.getCount(jlm+'_'+iPosition)/c);
                }
            }
            //System.out.println(feTCount);
            /*
                for (String englishWord : englishWords) {
                    String jPosition = Integer.toString(englishWords.indexOf(englishWord)); 
                    for (String frenchWord : frenchWords) {
                        String iPosition = Integer.toString(frenchWords.indexOf(frenchWord));
                        /*
                        double c = feTCount.getCounter(jPosition+'_'+iPosition).getCount(frenchWord+'_'+englishWord)/Z.getCount(englishWord);
                        double cc = efTCount.getCounter(iPosition+'_'+jPosition).getCount(englishWord+'_'+frenchWord)/Y.getCount(frenchWord);

                        double cL = feTCount.getCounter(jPosition+'_'+iPosition).getCount(frenchWord+'_'+englishWord)/ZL.getCount(frenchLength+'_'+englishLength+'_'+jPosition);
                        double ccL = efTCount.getCounter(iPosition+'_'+jPosition).getCount(englishWord+'_'+frenchWord)/YL.getCount(englishLength+'_'+frenchLength+'_'+iPosition);

                        //
                        double c = feTCount.getCounter(jPosition+'_'+iPosition).getCount(frenchWord+'_'+englishWord);
                        double cc = efTCount.getCounter(iPosition+'_'+jPosition).getCount(englishWord+'_'+frenchWord);
                        double cL = feTCount.getCounter(jPosition+'_'+iPosition).getCount(frenchWord+'_'+englishWord);
                        double ccL = efTCount.getCounter(iPosition+'_'+jPosition).getCount(englishWord+'_'+frenchWord);


                        feCount.incrementCount(frenchWord, englishWord, c);
                        fCount.incrementCount(frenchWord, c);
                        efCount.incrementCount(englishWord, frenchWord, cc);
                        eCount.incrementCount(englishWord, cc);

                        feLength.incrementCount(frenchLength+'_'+englishLength+'_'+jPosition, iPosition, cL);
                        fLength.incrementCount(frenchLength+'_'+englishLength+'_'+jPosition, cL);
                        efLength.incrementCount(englishLength+'_'+frenchLength+'_'+iPosition, jPosition, ccL);
                        eLength.incrementCount(englishLength+'_'+frenchLength+'_'+iPosition, ccL);

                    }
                }
            }
            for (SentencePair sentencePair : trainingSentencePairs) {
                List<String> frenchWords = sentencePair.getFrenchWords();
                List<String> englishWords = sentencePair.getEnglishWords();
                String frenchLength = Integer.toString(frenchWords.size());
                String englishLength = Integer.toString(englishWords.size());

                for (String frenchWord : frenchWords) {
                    double c = fCount.getCount(frenchWord);
                    String iPosition = Integer.toString(frenchWords.indexOf(frenchWord)); 
                    double ccL = eLength.getCount(frenchLength+'_'+englishLength+'_'+iPosition);
                    for (String englishWord : englishWords) {
                        double cc = eCount.getCount(englishWord);
                        String jPosition = Integer.toString(englishWords.indexOf(englishWord));
                        double cL = fLength.getCount(frenchLength+'_'+englishLength+'_'+jPosition);
                        feTCount.setCount(jPosition+'_'+iPosition, frenchWord+'_'+englishWord, (feCount.getCounter(frenchWord).getCount(englishWord)/c)
                                * (feLength.getCounter(frenchLength+'_'+englishLength+'_'+jPosition).getCount(iPosition)/cL));
                        efTCount.setCount(iPosition+'_'+jPosition, englishWord+'_'+frenchWord, (efCount.getCounter(englishWord).getCount(frenchWord)/cc)
                                * (efLength.getCounter(englishLength+'_'+frenchLength+'_'+iPosition).getCount(jPosition)/ccL));
                    }
                }
            }
            */
            iteration += 1;
            //if (iteration == 1) {System.out.println(efQCount);} 
            }
        }

        public Alignment alignSentencePair(SentencePair sentencePair) {
           /* 
            Alignment alignment = new Alignment();
            List<String> frenchWords = sentencePair.getFrenchWords();
            List<String> englishWords = sentencePair.getEnglishWords();
            int numFrenchWords = sentencePair.getFrenchWords().size();
            int numEnglishWords = sentencePair.getEnglishWords().size();
            String frenchLength = Integer.toString(numFrenchWords);
            String englishLength = Integer.toString(numEnglishWords);

            for (String frenchWord : frenchWords) {
                double bestp = Double.NEGATIVE_INFINITY;
                int bestj = -1;
                int frenchPosition = frenchWords.indexOf(frenchWord);
                String iPosition = Integer.toString(frenchPosition);
                for (String englishWord : englishWords) {
                    int englishPosition = englishWords.indexOf(englishWord);
                    String jPosition = Integer.toString(englishPosition);
                    
                    double IBM2Score  = efTCount.getCounter(englishWord).getCount(frenchWord)
                                * (feQCount.getCounter(frenchLength+'_'+englishLength+'_'+jPosition).getCount(iPosition))
                                + efTCount.getCounter(englishWord).getCount(frenchWord)
                                * (efQCount.getCounter(englishLength+'_'+frenchLength+'_'+iPosition).getCount(jPosition)) ;
                    if (IBM2Score > bestp) {
                        bestp = IBM2Score;
                        bestj = englishWords.indexOf(englishWord);
                    }
                }
                alignment.addAlignment(bestj, frenchPosition, true);
            }
            return alignment;
        }
        */
            Alignment alignment = new Alignment();
            int numFrenchWords = sentencePair.getFrenchWords().size();
            int numEnglishWords = sentencePair.getEnglishWords().size();
            String frenchLength = Integer.toString(numFrenchWords);
            String englishLength = Integer.toString(numEnglishWords);

            List<String> frenchWords = sentencePair.getFrenchWords();
            List<String> englishWords = sentencePair.getEnglishWords();
            List<Integer> frenchPositions = new ArrayList<Integer>();
            List<Integer> englishPositions = new ArrayList<Integer>();
            for (int i = 0; i < numFrenchWords; i++) {
               frenchPositions.add(i); 
            }
            for (int i = 0; i < numEnglishWords; i++) {
                englishPositions.add(i);
            }


            while (!frenchPositions.isEmpty() && !englishPositions.isEmpty()) {
                double maxScore = Double.NEGATIVE_INFINITY;
                int maxEnglishPosition = -1;
                int maxFrenchPosition = -1;
                int frenchPosition = -1;

                for (String frenchWord : frenchWords) {
                    frenchPosition += 1;
                    if (!frenchPositions.contains(frenchPosition)) {continue;}
                    String iPosition = Integer.toString(frenchPosition);
                    int englishPosition = -1;
                    for (String englishWord : englishWords) {
                        englishPosition += 1;
                        if (!englishPositions.contains(englishPosition)) {continue;}
                        String jPosition = Integer.toString(englishPosition);
                        double IBM2Score = feTCount.getCounter(frenchWord).getCount(englishWord)
                            *(feQCount.getCounter(frenchLength+'_'+englishLength+'_'+jPosition).getCount(iPosition))
                            + efTCount.getCounter(englishWord).getCount(frenchWord)
                            * (efQCount.getCounter(englishLength+'_'+frenchLength+'_'+iPosition).getCount(jPosition));
                        if (IBM2Score > maxScore) {
                            maxEnglishPosition = englishPosition;
                            maxFrenchPosition = frenchPosition;
                            maxScore = IBM2Score;
                        }
                    } 

                }
                if (maxEnglishPosition == -1 && maxFrenchPosition == -1) {break;}
                if (maxEnglishPosition != -1) {
                    englishPositions.remove(englishPositions.indexOf(maxEnglishPosition));
                    frenchPositions.remove(frenchPositions.indexOf(maxFrenchPosition));
                    alignment.addAlignment(maxEnglishPosition, maxFrenchPosition, true);
                }
            }
            for (int frenchPosition : frenchPositions) {
               alignment.addAlignment(-1, frenchPosition, true); 
            }
            return alignment;
        }
    
    }

    static class HMMWordAligner implements WordAligner {
        CounterMap<String, String> feTCount = new CounterMap<String, String>();
        CounterMap<String, String> efTCount = new CounterMap<String, String>();
        CounterMap<String, String> feQCount  = new CounterMap<String, String>();
        CounterMap<String, String> efQCount = new CounterMap<String, String>();
        int flag = 0;

        public void train(List<SentencePair> trainingSentencePairs) {
            int iteration = 0;
           
            for (SentencePair sentencePair : trainingSentencePairs) {
                List<String> frenchWords = sentencePair.getFrenchWords();
                List<String> englishWords = sentencePair.getEnglishWords();
                for (String englishWord : englishWords) {
                    int englishNum = englishWords.size();
                    for (String frenchWord : frenchWords) {
                        int frenchNum = frenchWords.size();
                        feTCount.setCount(frenchWord, englishWord, 1.0/(1+englishNum));
                        efTCount.setCount(englishWord, frenchWord, 1.0/(1+frenchNum));
                    }
                }
            }
/*
	        for (SentencePair sentencePair : trainingSentencePairs) {
	            List<String> frenchWords = sentencePair.getFrenchWords();
	            List<String> englishWords = sentencePair.getEnglishWords();
	            int frenchNum = frenchWords.size();
	            int englishNum = englishWords.size();
	            String fLength = Integer.toString(frenchNum);
	            String eLength = Integer.toString(englishNum);
	    
	            for (int jj = 0; jj < englishNum; jj++) {
	                double Total = 0.0;
	                String jjLength = Integer.toString(jj);
	                for (int j = 0; j < englishNum; j++) {
                        if (j != jj){
		                    Total += Math.pow(englishNum - Math.abs(j - jj), 2);
	                    }
                        else{
                            Total += Math.pow(englishNum - Math.abs(j - jj) -2, 2);
                        }
                    }
	                for (int j = 0; j < englishNum; j++) {
		                String jLength = Integer.toString(j);
	                    if (j != jj){
                            efQCount.incrementCount(jjLength+'_'+eLength, jLength, Math.pow(englishNum - Math.abs(j-jj), 2)/Total);
	                    }
                        else {
                            efQCount.incrementCount(jjLength+'_'+eLength, jLength, Math.pow(englishNum - Math.abs(j-jj) -2, 2)/Total);
                        }
                    }
	            }
	            for (int ii = 0; ii < frenchNum; ii++) {
	                double Total = 0.0;
	                String iiLength = Integer.toString(ii);
	                for (int i = 0; i < frenchNum; i++) {
		                Total += frenchNum - Math.abs(i - ii);
	                }
	                for (int i = 0; i < frenchNum; i++) {
		                String iLength = Integer.toString(i);
	                    feQCount.incrementCount(iiLength+'_'+fLength, iLength, (frenchNum - Math.abs(i-ii))/Total);
	                }
	            }
	        }
*/
            while (!(iteration == 5)){
            CounterMap<String, String> feCount = new CounterMap<String, String>();
            Counter<String> fCount = new Counter<String>();

            CounterMap<String, String> efCount = new CounterMap<String, String>();
            Counter<String> eCount = new Counter<String>();
            
            for (SentencePair sentencePair : trainingSentencePairs) {
                List<String> frenchWords = sentencePair.getFrenchWords();
                List<String> englishWords = sentencePair.getEnglishWords();
                
                
                for (String englishWord : englishWords) {
                    double Z = 0.0;
                    for (String frenchWord : frenchWords) {
                        Z += feTCount.getCounter(frenchWord).getCount(englishWord);
                    }
                    for (String frenchWord: frenchWords) {
                        feCount.incrementCount(frenchWord, englishWord, feTCount.getCounter(frenchWord).getCount(englishWord)/Z);
                        fCount.incrementCount(frenchWord, feTCount.getCounter(frenchWord).getCount(englishWord)/Z);
                    }
                }
                
                for (String frenchWord : frenchWords) {
                    double Y = 0.0;
                    for (String englishWord : englishWords) {
                        Y += efTCount.getCounter(englishWord).getCount(frenchWord);
                    }
                    for (String englishWord : englishWords) {
                        efCount.incrementCount(englishWord, frenchWord, efTCount.getCounter(englishWord).getCount(frenchWord)/Y);
                        eCount.incrementCount(englishWord, efTCount.getCounter(englishWord).getCount(frenchWord)/Y);
                    }
                }
            }
            for (String frenchWord : feTCount.keySet()) {
                double c = fCount.getCount(frenchWord);
                for (String englishWord : feTCount.getCounter(frenchWord).keySet()) {
                    double cc = eCount.getCount(englishWord);
                    feTCount.setCount(frenchWord, englishWord, feCount.getCounter(frenchWord).getCount(englishWord)/c);
                    efTCount.setCount(englishWord, frenchWord, efCount.getCounter(englishWord).getCount(frenchWord)/cc);
                }
            }
            iteration += 1;
            }
            //System.out.println(efQCount);
        }

        public double Viterbi (int j, int i, SentencePair sentencePair, Map<String, Integer> BP, Map<String, Double> VT){
            double maxScore = Double.NEGATIVE_INFINITY;
            int maxj = j;
            List<String> englishWords = sentencePair.getEnglishWords();
            List<String> frenchWords = sentencePair.getFrenchWords();
            int englishNum = englishWords.size();
            int frenchNum = frenchWords.size();
            String iiS = Integer.toString(i-1);
            String J = Integer.toString(englishNum);
            String jS = Integer.toString(j);
            String iS = Integer.toString(i);
            String englishWord = new String();
            String frenchWord = new String();
            double prob = 0.0;
            
            englishWord = englishWords.get(j);
            frenchWord = frenchWords.get(i);
            
            prob = efTCount.getCounter(englishWord).getCount(frenchWord)
                        + feTCount.getCounter(frenchWord).getCount(englishWord);
           /* 
            if (prob == 0 && feTCount.keySet().contains(frenchWord) && efTCount.keySet().contains(englishWord)){
                System.out.println(efTCount.getCounter(englishWord).getCount(frenchWord));
                //System.out.println(feTCount.getCounter(frenchWord));
                System.out.println("**********");
            }
            //
            if (prob == 0 && feTCount.keySet().contains(frenchWord) && feTCount.getCounter(frenchWord).keySet().contains(englishWord)){
                System.out.println(efTCount.getCounter(englishWord).getCount(frenchWord));
            }
            */

            if (i == 0) {
                if (!feTCount.keySet().contains(frenchWord)){
                //if (prob == 0){
                    maxScore = 1;
                }
                else {
                    maxScore = prob;
                }
                maxj = -1;
            }
            
            else{
                if (prob == 0){
                    prob = 1;
                }
                for (int jj = 0; jj < englishNum - 1; jj++){
                    double score = 0.0;
                    String jjS = Integer.toString(jj);
                    if (VT.keySet().contains(jjS+'_'+iiS)) {
                        score = VT.get(jjS+'_'+iiS)*prob*Math.pow(efQCount.getCounter(jjS+'_'+J).getCount(jS), 1.5);
                    }
                    else {
                        score = Viterbi(jj, i-1, sentencePair, BP, VT)*prob*Math.pow(efQCount.getCounter(jjS+'_'+J).getCount(jS), 1.5);
                        
                    }
                    if (score > maxScore || maxScore == Double.NEGATIVE_INFINITY) {
                        maxScore = score;
                        maxj = jj;
                    }
                    
                }
            }
           /*
            if (maxScore == 0 && maxj != -1) {
                System.out.println(englishWord);
                System.out.println(maxj);
            }
          */
            if (prob == 1 && feTCount.keySet().contains(frenchWord)){
                maxScore = 0;
            }
            BP.put(jS+'_'+iS, maxj);
            VT.put(jS+'_'+iS, maxScore);
            return maxScore;
        }

        public Alignment alignSentencePair(SentencePair sentencePair) {
           
            Map<String, Integer> BP = new HashMap<String, Integer>();
            Map<String, Double> VT = new HashMap<String, Double>();
            
            Alignment alignment = new Alignment();
            List<String> frenchWords = sentencePair.getFrenchWords();
            List<String> englishWords = sentencePair.getEnglishWords();
            int frenchNum = frenchWords.size();
            int englishNum = englishWords.size();
            List<Integer> frenchPositions = new ArrayList<Integer>();
            List<Integer> englishPositions = new ArrayList<Integer>();
            for (int i = 0; i < frenchNum; i++) {
               frenchPositions.add(i); 
            }
            for (int i = 0; i < englishNum; i++) {
                englishPositions.add(i);
            }

	            String fLength = Integer.toString(frenchNum);
	            String eLength = Integer.toString(englishNum);
	    
	            for (int jj = 0; jj < englishNum; jj++) {
	                double Total = 0.0;
	                String jjLength = Integer.toString(jj);
	                for (int j = 0; j < englishNum; j++) {
                        if (j != jj){
		                    Total += Math.pow(englishNum - Math.abs(j - jj), 2);
	                    }
                        else{
                            if (englishNum == 2){
                                Total += Math.pow(englishNum - Math.abs(j - jj) -1, 2);
                            }
                            else {
                                Total += Math.pow(englishNum - Math.abs(j - jj) -2, 2);
                            }
                        }
                    }
	                for (int j = 0; j < englishNum; j++) {
		                String jLength = Integer.toString(j);
	                    if (j != jj){
                            efQCount.incrementCount(jjLength+'_'+eLength, jLength, Math.pow(englishNum - Math.abs(j-jj), 2)/Total);
	                    }
                        else {
                            if (englishNum == 2){
                                efQCount.incrementCount(jjLength+'_'+eLength, jLength, Math.pow(englishNum - Math.abs(j-jj) -1, 2)/Total);
                            }
                            else {
                                efQCount.incrementCount(jjLength+'_'+eLength, jLength, Math.pow(englishNum - Math.abs(j-jj) -2, 2)/Total);
                            }
                        }
                    }
	            }


            //double score = Viterbi(-1, frenchNum - 1, sentencePair, BP, VT);
            //System.out.println("***********");
            String tmp = Integer.toString(frenchNum -1);
            //int bestj = BP.get("-1"+"_"+tmp);
            int bestj = englishNum - 1;
            flag = 0;
            double score = Viterbi(bestj, frenchNum - 1, sentencePair, BP, VT);
            if (flag == 0){
                alignment.addAlignment(bestj, frenchNum-1, true);
                for (int i = frenchNum-1; i > 0; i--) {
                    tmp = Integer.toString(bestj);
                    String tmp2 = Integer.toString(i);
                    bestj = BP.get(tmp+"_"+tmp2);
                    alignment.addAlignment(bestj, i - 1, true);
                }
            }
            else {
                System.out.println("unknown");
            while (!frenchPositions.isEmpty() && !englishPositions.isEmpty()) {
                double maxScore = Double.NEGATIVE_INFINITY;
                int maxEnglishPosition = -1;
                int maxFrenchPosition = -1;
                int frenchPosition = -1;

                for (String frenchWord : frenchWords) {
                    frenchPosition += 1;
                    if (!frenchPositions.contains(frenchPosition)) {continue;}
                    int englishPosition = -1;
                    for (String englishWord : englishWords) {
                        englishPosition += 1;
                        if (!englishPositions.contains(englishPosition)) {continue;}
                        double IBM1Score = (feTCount.getCounter(frenchWord).getCount(englishWord)
                            + efTCount.getCounter(englishWord).getCount(frenchWord));
                        if (IBM1Score > maxScore) {
                            maxEnglishPosition = englishPosition;
                            maxFrenchPosition = frenchPosition;
                            maxScore = IBM1Score;
                        }
                    } 

                }
                if (maxEnglishPosition == -1 && maxFrenchPosition == -1) {break;}
                if (maxEnglishPosition != -1) {
                    englishPositions.remove(englishPositions.indexOf(maxEnglishPosition));
                    frenchPositions.remove(frenchPositions.indexOf(maxFrenchPosition));
                    alignment.addAlignment(maxEnglishPosition, maxFrenchPosition, true);
                }
            }
            for (int frenchPosition : frenchPositions) {
               alignment.addAlignment(-1, frenchPosition, true); 
            }
                
            }
            return alignment;
        }
 
}
    static class HMMIBM2WordAligner implements WordAligner {

        CounterMap<String, String> efTCount = new CounterMap<String, String>();
        CounterMap<String, String> efQCount  = new CounterMap<String, String>();
	    CounterMap<String, String> efQQCount = new CounterMap<String, String>();
        //CounterMap<String, String> feTCount = new CounterMap<String, String>();
        //CounterMap<String, String> feQQCount = new CounterMap<String, String>();

        public void train(List<SentencePair> trainingSentencePairs) {
            int iteration = 0;
           
            for (SentencePair sentencePair : trainingSentencePairs) {
                List<String> frenchWords = sentencePair.getFrenchWords();
                List<String> englishWords = sentencePair.getEnglishWords();
		        int englishNum = englishWords.size();
		        String englishLength = Integer.toString(englishNum);
		        int frenchNum = frenchWords.size();
		        String frenchLength = Integer.toString(frenchNum);
		        int englishPosition = -1;
                for (String englishWord : englishWords) {
		            englishPosition += 1;
		            String jPosition = Integer.toString(englishPosition);
		            int frenchPosition = -1;
                    for (String frenchWord : frenchWords) {
			            frenchPosition += 1;
			            String  iPosition = Integer.toString(frenchPosition);
                        efTCount.setCount(englishWord, frenchWord, 1.0/(1+frenchNum));
			            efQQCount.setCount(englishLength+'_'+frenchLength+'_'+iPosition, jPosition, 1.0/(1+englishNum));
                        //feTCount.setCount(frenchWord, englishWord, 1.0/(1+frenchNum));
                        //feQQCount.setCount(frenchLength+'_'+englishLength+'_'+jPosition, iPosition, 1.0/(1+frenchNum));
                    }
                }
            }

	        for (SentencePair sentencePair : trainingSentencePairs) {
	            List<String> frenchWords = sentencePair.getFrenchWords();
	            List<String> englishWords = sentencePair.getEnglishWords();
	            int frenchNum = frenchWords.size();
	            int englishNum = englishWords.size();
	            String fLength = Integer.toString(frenchNum);
	            String eLength = Integer.toString(englishNum);
	    
	            for (int jj = 0; jj < englishNum; jj++) {
	                double Total = 0.0;
	                String jjLength = Integer.toString(jj);
	                for (int j = 0; j < englishNum; j++) {
		                Total += englishNum - Math.abs(j - jj);
	                }
	                for (int j = 0; j < englishNum; j++) {
		                String jLength = Integer.toString(j);
	                    efQCount.incrementCount(jjLength+'_'+eLength, jLength, (englishNum - Math.abs(j-jj))/Total);
	                }
	            }
            }
            while (!(iteration == 5)){

            Counter<String> efCount = new Counter<String>();
            Counter<String> eCount = new Counter<String>();

            Counter<String> efLength = new Counter<String>();
            Counter<String> eLength = new Counter<String>();
/*
            Counter<String> feCount = new Counter<String>();
            Counter<String> fCount = new Counter<String>();

            Counter<String> feLength = new Counter<String>();
            Counter<String> fLength = new Counter<String>();
*/
            for (SentencePair sentencePair : trainingSentencePairs) {
                List<String> frenchWords = sentencePair.getFrenchWords();
                List<String> englishWords = sentencePair.getEnglishWords();
                String frenchLength = Integer.toString(frenchWords.size());
                String englishLength = Integer.toString(englishWords.size());
/*
                int englishPosition = -1;
                for (String englishWord : englishWords) {
                    double Z = 0.0;
                    englishPosition += 1;
                    String jPosition = Integer.toString(englishPosition);
                    int frenchPosition = -1;
                    for (String frenchWord : frenchWords) {
                        frenchPosition += 1;
                        String iPosition = Integer.toString(frenchPosition);
                        Z += feQQCount.getCounter(frenchLength+'_'+englishLength+'_'+jPosition).getCount(iPosition)
                                * feTCount.getCounter(frenchWord).getCount(englishWord);
                    }
                    frenchPosition = -1;
                    for (String frenchWord : frenchWords) {
                        frenchPosition += 1;
                        String iPosition = Integer.toString(frenchPosition);
                        double c = feQQCount.getCounter(frenchLength+'_'+englishLength+'_'+jPosition).getCount(iPosition)
                                    * feTCount.getCounter(frenchWord).getCount(englishWord)/Z;
                        feCount.incrementCount(frenchWord+'_'+englishWord, c);
                        fCount.incrementCount(frenchWord, c);
                        feLength.incrementCount(frenchLength+'_'+englishLength+'_'+jPosition+'_'+iPosition, c);
                        fLength.incrementCount(frenchLength+'_'+englishLength+'_'+jPosition, c);
                    }
                }
*/
                int frenchPosition = -1;
                for (String frenchWord : frenchWords) {
                    double Y = 0.0;
                    frenchPosition += 1;
                    String iPosition = Integer.toString(frenchPosition);
                    int englishPosition = -1;
                    for (String englishWord : englishWords) {
                        englishPosition += 1;
                        String jPosition = Integer.toString(englishPosition);
                        Y += efQQCount.getCounter(englishLength+'_'+frenchLength+'_'+iPosition).getCount(jPosition)
				            * efTCount.getCounter(englishWord).getCount(frenchWord);
                    }
		            englishPosition = -1;
                    for (String englishWord : englishWords) {
                        englishPosition += 1;
                        String jPosition = Integer.toString(englishPosition);
                        double c = efQQCount.getCounter(englishLength+'_'+frenchLength+'_'+iPosition).getCount(jPosition)
                                    * efTCount.getCounter(englishWord).getCount(frenchWord)/Y;
                        efCount.incrementCount(englishWord+'_'+frenchWord, c);
                        eCount.incrementCount(englishWord, c);
                        efLength.incrementCount(englishLength+'_'+frenchLength+'_'+iPosition+'_'+jPosition, c);
                        eLength.incrementCount(englishLength+'_'+frenchLength+'_'+iPosition, c);                    
		            }
                }
            }
            for (String englishWord : efTCount.keySet()) {
                double c = eCount.getCount(englishWord);
                for (String frenchWord : efTCount.getCounter(englishWord).keySet()) {
                     efTCount.setCount(englishWord,frenchWord, efCount.getCount(englishWord+'_'+frenchWord)/c);
                }
            }
            for (String ilm : efQQCount.keySet()) {
                double c = eLength.getCount(ilm);
                for (String jPosition : efQQCount.getCounter(ilm).keySet()) {
                    efQQCount.setCount(ilm, jPosition, efLength.getCount(ilm+'_'+jPosition)/c);
                }
            }
            /*
            for (String jlm : feQQCount.keySet()) {
                double c = fLength.getCount(jlm);
                for (String iPosition : feQQCount.getCounter(jlm).keySet()) {
                    feQQCount.setCount(jlm, iPosition, feLength.getCount(jlm+'_'+iPosition)/c);
                }
            }
	        */
            iteration += 1;
            }

        }

        public double Viterbi (int j, int i, SentencePair sentencePair, Map<String, Integer> BP, Map<String, Double> VT){
            double maxScore = Double.NEGATIVE_INFINITY;
            int maxj = j;
            List<String> englishWords = sentencePair.getEnglishWords();
            List<String> frenchWords = sentencePair.getFrenchWords();
            int englishNum = englishWords.size();
            int frenchNum = frenchWords.size();
	        String englishLength = Integer.toString(englishNum);
	        String frenchLength = Integer.toString(frenchNum);
            String iiS = Integer.toString(i-1);
            String J = Integer.toString(englishNum);
            String jS = Integer.toString(j);
            String iS = Integer.toString(i);
            String englishWord = new String();
            String frenchWord = new String();
            double prob = 0.0;
            if (j != -1){
                englishWord = englishWords.get(j);
                frenchWord = frenchWords.get(i); 
                prob = efTCount.getCounter(englishWord).getCount(frenchWord)
			            *efQQCount.getCounter(englishLength+'_'+frenchLength+'_'+iS).getCount(jS);
			//+feTCount.getCounter(frenchWord).getCount(englishWord)
			//	*feQQCount.getCounter(frenchLength+'_'+englishLength+'_'+jS).getCount(iS);
            }
            if (i == 0) {
                maxScore = prob;
                maxj = -1;
            }
            else{
                for (int jj = 0; jj < englishNum; jj++){
                    double score = 0.0;
                    String jjS = Integer.toString(jj);
                    /*
                    if (j == -1) {
                        //System.out.println("Helloe!");
                        score = Viterbi(jj, i, sentencePair, BP, VT);
                    }
                    */
                    if (VT.keySet().contains(jjS+'_'+iiS)) {
                        score = VT.get(jjS+'_'+iiS)*prob*Math.pow(efQCount.getCounter(jjS+'_'+J).getCount(jS),0.1);
                    }
                    else {
                        score = Viterbi(jj, i-1, sentencePair, BP, VT)*prob*Math.pow(efQCount.getCounter(jjS+'_'+J).getCount(jS),0.1);
                    }
                    if (score > maxScore || maxScore == Double.NEGATIVE_INFINITY) {
                        maxScore = score;
                        maxj = jj;
                    }
                }
            }
            BP.put(jS+'_'+iS, maxj);
            VT.put(jS+'_'+iS, maxScore);
            return maxScore;
        }

        public Alignment alignSentencePair(SentencePair sentencePair) {
           
            Map<String, Integer> BP = new HashMap<String, Integer>();
            Map<String, Double> VT = new HashMap<String, Double>();
            
            Alignment alignment = new Alignment();
            List<String> frenchWords = sentencePair.getFrenchWords();
            List<String> englishWords = sentencePair.getEnglishWords();
            int frenchNum = frenchWords.size();
            int englishNum = englishWords.size();
            //double score = Viterbi(-1, frenchNum - 1, sentencePair, BP, VT);
            String tmp = Integer.toString(frenchNum -1);
            //int bestj = BP.get("-1"+"_"+tmp);
            int bestj = englishNum - 1; 
            double score = Viterbi(bestj, frenchNum - 1, sentencePair, BP, VT);
            alignment.addAlignment(bestj, frenchNum-1, true);
            for (int i = frenchNum-1; i > 0; i--) {
                tmp = Integer.toString(bestj);
                String tmp2 = Integer.toString(i);
                bestj = BP.get(tmp+"_"+tmp2);
                alignment.addAlignment(bestj, i - 1, true);
            }
            
            return alignment;
        }
 
}



  public static void main(String[] args) throws IOException {
    // Parse command line flags and arguments
    Map<String,String> argMap = CommandLineUtils.simpleCommandLineParser(args);

    // Set up default parameters and settings
    String basePath = ".";
    int maxTrainingSentences = 0;
    boolean verbose = false;
    String dataset = "mini";
    String model = "baseline";

    // Update defaults using command line specifications
    if (argMap.containsKey("-path")) {
      basePath = argMap.get("-path");
      System.out.println("Using base path: "+basePath);
    }
    if (argMap.containsKey("-sentences")) {
      maxTrainingSentences = Integer.parseInt(argMap.get("-sentences"));
      System.out.println("Using an additional "+maxTrainingSentences+" training sentences.");
    }
    if (argMap.containsKey("-data")) {
      dataset = argMap.get("-data");
      System.out.println("Running with data: "+dataset);
    } else {
      System.out.println("No data set specified.  Use -data [miniTest, validate, test].");
    }
    if (argMap.containsKey("-model")) {
      model = argMap.get("-model");
      System.out.println("Running with model: "+model);
    } else {
      System.out.println("No model specified.  Use -model modelname.");
    }
    if (argMap.containsKey("-verbose")) {
      verbose = true;
    }

    // Read appropriate training and testing sets.
    List<SentencePair> trainingSentencePairs = new ArrayList<SentencePair>();
    if (! dataset.equals("miniTest") && maxTrainingSentences > 0)
      trainingSentencePairs = readSentencePairs(basePath+"/training", maxTrainingSentences);
    List<SentencePair> testSentencePairs = new ArrayList<SentencePair>();
    Map<Integer,Alignment> testAlignments = new HashMap<Integer, Alignment>();
    if (dataset.equalsIgnoreCase("validate")) {
      testSentencePairs = readSentencePairs(basePath+"/trial", Integer.MAX_VALUE);
      testAlignments = readAlignments(basePath+"/trial/trial.wa");
    } else if (dataset.equalsIgnoreCase("miniTest")) {
      testSentencePairs = readSentencePairs(basePath+"/mini", Integer.MAX_VALUE);
      testAlignments = readAlignments(basePath+"/mini/mini.wa");
    } else {
      throw new RuntimeException("Bad data set mode: "+ dataset+", use validate or miniTest.");
    }
    trainingSentencePairs.addAll(testSentencePairs);

    // Build model
    WordAligner wordAligner = null;
    if (model.equalsIgnoreCase("baseline")) {
      wordAligner = new BaselineWordAligner();
    }
    // TODO : build other alignment models
        else if (model.equalsIgnoreCase("Dice")){
            wordAligner = new DiceWordAligner();
            wordAligner.train(trainingSentencePairs);
        }
	else if (model.equalsIgnoreCase("IBM1")) {
            wordAligner = new IBM1WordAligner();
            wordAligner.train(trainingSentencePairs);
        }
	else if (model.equalsIgnoreCase("IBM2")) {
            wordAligner = new IBM2WordAligner();
            wordAligner.train(trainingSentencePairs);
        }
    else if (model.equalsIgnoreCase("HMM")) {
        wordAligner = new HMMWordAligner();
        wordAligner.train(trainingSentencePairs);
    }
    else if (model.equalsIgnoreCase("HMMIBM2")) {
        wordAligner = new HMMIBM2WordAligner();
        wordAligner.train(trainingSentencePairs);
    }

    // Test model
    test(wordAligner, testSentencePairs, testAlignments, verbose);
    
    // Generate file for submission
    testSentencePairs = readSentencePairs(basePath+"/test", Integer.MAX_VALUE);
    predict(wordAligner, testSentencePairs, basePath+"/"+model+".out");
  }

  private static void test(WordAligner wordAligner, List<SentencePair> testSentencePairs, Map<Integer, Alignment> testAlignments, boolean verbose) {
    int proposedSureCount = 0;
    int proposedPossibleCount = 0;
    int sureCount = 0;
    int proposedCount = 0;
    for (SentencePair sentencePair : testSentencePairs) {
      System.out.println(sentencePair.getEnglishWords());
      Alignment proposedAlignment = wordAligner.alignSentencePair(sentencePair);
      Alignment referenceAlignment = testAlignments.get(sentencePair.getSentenceID());
      if (referenceAlignment == null)
        throw new RuntimeException("No reference alignment found for sentenceID "+sentencePair.getSentenceID());
      if (verbose) System.out.println("Alignment:\n"+Alignment.render(referenceAlignment,proposedAlignment,sentencePair));
      for (int frenchPosition = 0; frenchPosition < sentencePair.getFrenchWords().size(); frenchPosition++) {
        for (int englishPosition = 0; englishPosition < sentencePair.getEnglishWords().size(); englishPosition++) {
          boolean proposed = proposedAlignment.containsSureAlignment(englishPosition, frenchPosition);
          boolean sure = referenceAlignment.containsSureAlignment(englishPosition, frenchPosition);
          boolean possible = referenceAlignment.containsPossibleAlignment(englishPosition, frenchPosition);
          if (proposed && sure) proposedSureCount += 1;
          if (proposed && possible) proposedPossibleCount += 1;
          if (proposed) proposedCount += 1;
          if (sure) sureCount += 1;
        }
      }
    }
    System.out.println("Precision: "+proposedPossibleCount/(double)proposedCount);
    System.out.println("Recall: "+proposedSureCount/(double)sureCount);
    System.out.println("AER: "+(1.0-(proposedSureCount+proposedPossibleCount)/(double)(sureCount+proposedCount)));
  }

  private static void predict(WordAligner wordAligner, List<SentencePair> testSentencePairs, String path) throws IOException {
	BufferedWriter writer = new BufferedWriter(new FileWriter(path));
    for (SentencePair sentencePair : testSentencePairs) {
      Alignment proposedAlignment = wordAligner.alignSentencePair(sentencePair);
      for (int frenchPosition = 0; frenchPosition < sentencePair.getFrenchWords().size(); frenchPosition++) {
        for (int englishPosition = 0; englishPosition < sentencePair.getEnglishWords().size(); englishPosition++) {
          if (proposedAlignment.containsSureAlignment(englishPosition, frenchPosition)) {
        	writer.write(frenchPosition + "-" + englishPosition + " ");
          }
        }
      }
      writer.write("\n");
    }
    writer.close();
  }

  // BELOW HERE IS IO CODE

  private static Map<Integer, Alignment> readAlignments(String fileName) {
    Map<Integer,Alignment> alignments = new HashMap<Integer, Alignment>();
    try {
      BufferedReader in = new BufferedReader(new FileReader(fileName));
      while (in.ready()) {
        String line = in.readLine();
        String[] words = line.split("\\s+");
        if (words.length != 4)
          throw new RuntimeException("Bad alignment file "+fileName+", bad line was "+line);
        Integer sentenceID = Integer.parseInt(words[0]);
        Integer englishPosition = Integer.parseInt(words[1])-1;
        Integer frenchPosition = Integer.parseInt(words[2])-1;
        String type = words[3];
        Alignment alignment = alignments.get(sentenceID);
        if (alignment == null) {
          alignment = new Alignment();
          alignments.put(sentenceID, alignment);
        }
        alignment.addAlignment(englishPosition, frenchPosition, type.equals("S"));
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return alignments;
  }

  private static List<SentencePair> readSentencePairs(String path, int maxSentencePairs) {
    List<SentencePair> sentencePairs = new ArrayList<SentencePair>();
    List<String> baseFileNames = getBaseFileNames(path);
    for (String baseFileName : baseFileNames) {
      if (sentencePairs.size() >= maxSentencePairs)
        continue;
      sentencePairs.addAll(readSentencePairs(baseFileName));
    }
    return sentencePairs;
  }

  private static List<SentencePair> readSentencePairs(String baseFileName) {
    List<SentencePair> sentencePairs = new ArrayList<SentencePair>();
    String englishFileName = baseFileName + "." + ENGLISH_EXTENSION;
    String frenchFileName = baseFileName + "." + FRENCH_EXTENSION;
    try {
      BufferedReader englishIn = new BufferedReader(new FileReader(englishFileName));
      BufferedReader frenchIn = new BufferedReader(new FileReader(frenchFileName));
      while (englishIn.ready() && frenchIn.ready()) {
        String englishLine = englishIn.readLine();
        String frenchLine = frenchIn.readLine();
        Pair<Integer,List<String>> englishSentenceAndID = readSentence(englishLine);
        Pair<Integer,List<String>> frenchSentenceAndID = readSentence(frenchLine);
        if (! englishSentenceAndID.getFirst().equals(frenchSentenceAndID.getFirst()))
          throw new RuntimeException("Sentence ID confusion in file "+baseFileName+", lines were:\n\t"+englishLine+"\n\t"+frenchLine);
        sentencePairs.add(new SentencePair(englishSentenceAndID.getFirst(), baseFileName, englishSentenceAndID.getSecond(), frenchSentenceAndID.getSecond()));
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return sentencePairs;
  }

  private static Pair<Integer, List<String>> readSentence(String line) {
    int id = -1;
    List<String> words = new ArrayList<String>();
    String[] tokens = line.split("\\s+");
    for (int i = 0; i < tokens.length; i++) {
      String token = tokens[i];
      if (token.equals("<s")) continue;
      if (token.equals("</s>")) continue;
      if (token.startsWith("snum=")) {
        String idString = token.substring(5,token.length()-1);
        id = Integer.parseInt(idString);
        continue;
      }
      words.add(token.intern());
    }
    return new Pair<Integer, List<String>>(id, words);
  }

  private static List<String> getBaseFileNames(String path) {
    List<File> englishFiles = IOUtils.getFilesUnder(path, new FileFilter() {
      public boolean accept(File pathname) {
        if (pathname.isDirectory())
          return true;
        String name = pathname.getName();
        return name.endsWith(ENGLISH_EXTENSION);
      }
    });
    List<String> baseFileNames = new ArrayList<String>();
    for (File englishFile : englishFiles) {
      String baseFileName = chop(englishFile.getAbsolutePath(), "."+ENGLISH_EXTENSION);
      baseFileNames.add(baseFileName);
    }
    return baseFileNames;
  }

  private static String chop(String name, String extension) {
    if (! name.endsWith(extension)) return name;
    return name.substring(0, name.length()-extension.length());
  }

}
