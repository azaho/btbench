import Link from "next/link";

export default function Home() {
  return (
    <div className="min-h-screen bg-gray-900 flex flex-col items-center relative">
      
      {/* Centered Container */}
      <div className="absolute top-1/4 transform -translate-y-1/4 bg-gray-800 p-16 rounded-3xl shadow-2xl max-w-4xl text-center">
        
        {/* Background Image (Optional) */}
        <div className="absolute inset-0 opacity-10 bg-[url('/brain-pattern.png')] bg-cover bg-center rounded-3xl"></div>

        <h1 className="text-6xl font-extrabold text-white relative z-10">Brain Treebank</h1>
        
        <p className="text-gray-300 mt-6 text-2xl leading-relaxed relative z-10">
          Explore neural decoding leaderboards and analyze large-scale 
          intracranial brain recordings. Compare machine learning models 
          on structured language datasets derived from the Brain Treebank Dataset at MIT.
        </p>

        {/* Bigger Buttons */}
        <div className="flex justify-center gap-6 mt-10 relative z-10">
          <Link href="/leaderboards">
            <button className="w-64 h-16 text-2xl bg-green-500 hover:bg-green-600 text-white font-bold rounded-xl">
              Go to Leaderboards
            </button>
          </Link>
          <Link href="/about">
            <button className="w-64 h-16 text-2xl bg-blue-500 hover:bg-blue-600 text-white font-bold rounded-xl">
              About the Dataset
            </button>
          </Link>
        </div>
      </div>
    </div>
  );
}
