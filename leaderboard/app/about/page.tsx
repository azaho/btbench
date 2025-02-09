export default function About() {
    return (
        <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-start pt-12 p-6">
        <h1 className="text-4xl font-bold mb-6">About the Brain Treebank</h1>
        
        <div className="max-w-3xl text-center text-lg text-gray-300 leading-relaxed">
          <p>
            We present the <span className="font-semibold text-white">Brain Treebank</span>, a large-scale dataset of electrophysiological neural responses, 
            recorded from intracranial probes while 10 subjects watched one or more Hollywood movies. 
            Subjects watched on average 2.6 Hollywood movies, for an average viewing time of 4.3 hours, and a total of 43 hours. 
            The audio track for each movie was transcribed with manual corrections. Word onsets were manually annotated on spectrograms 
            of the audio track for each movie.
          </p>
  
          <p className="mt-4">
            Each transcript was automatically parsed and manually corrected into the <span className="font-semibold text-white">
            universal dependencies (UD) formalism</span>, assigning a part of speech to every word and a dependency parse to every sentence. 
            In total, subjects heard over 38,000 sentences (223,000 words), while they had on average 168 electrodes implanted.
          </p>
  
          <p className="mt-4">
            This is the largest dataset of intracranial recordings featuring <span className="font-semibold text-white">grounded naturalistic language</span>, 
            one of the largest English UD treebanks in general, and one of only a few UD treebanks aligned to multimodal features. 
            We hope that this dataset serves as a bridge between linguistic concepts, perception, and their neural representations.
          </p>
  
          <p className="mt-4">
            To that end, we present an analysis of which electrodes are sensitive to language features while also mapping out a rough 
            time course of language processing across these electrodes.
          </p>
        </div>
      </div>
    );
  }
  