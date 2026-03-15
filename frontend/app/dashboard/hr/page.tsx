export default function HRConsultantDashboard() {
  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold mb-4">HR Consultant Dashboard</h1>
      <p>View aggregated candidate data and leverage the Agentic Job Search AI to generate X-Ray searches for talent sourcing.</p>
      <div className="mt-4 p-4 border rounded">
        <h2 className="text-xl font-semibold">Generate Sourcing Queries</h2>
        <input type="text" placeholder="Role (e.g., Software Engineer)" className="mt-2 p-2 border rounded w-full" />
        <button className="mt-2 px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600">Generate X-Ray Queries</button>
      </div>
    </div>
  );
}
